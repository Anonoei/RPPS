"""Where a profile's parallel region starts, and whether it can (README.dev.md, Parallelism)

A profile marks the first step of the region: `"parallel": "<step id>"`. Everything from there on runs frame by frame, and
the steps before it (the serial front) find the frames and decide their parameters. The loader refuses a region that
could not give the same result as the serial runner:

  * every step in it has frame scope (`SCOPE`): nothing carried from one frame to the next. An operator that is frame
    scoped only when a control restarts it (`SCOPE_NEEDS`, a scrambler's `length`) needs the profile's controls to drive that;
  * no header in it publishes: a publisher in the region would set parameters of frames already being processed, so a
    header sits in the front and the region takes its resolved fields;
  * no change reaches back across the boundary: a late change (a step driven by a header after it) needs a re-run, which
    only the serial runner does.
"""
from dataclasses import dataclass

from ..base.rpps import FRAME
from ..errors import ConfigError

@dataclass
class Region:
    """first: index in the profile's steps of the first parallel step; steps: the region's steps;
    driven: {step id: {parameter}} the front resolves for every frame (what controls drive in the region)"""
    first: int
    steps: list
    driven: dict

    def __repr__(self):
        return f"<Region from {self.steps[0].id}: " + " ".join(s.id for s in self.steps) + ">"

def scope_of(step, driven=()):
    """FRAME when every receive operator of the step is frame scoped (given the parameters a control drives), else STREAM"""
    ops = [op for _, op in step.rx]
    why = []
    for op in ops:
        if getattr(op, "SCOPE", None) != FRAME:
            why.append(f"{type(op).__name__} keeps state across frames")
        else:
            miss = [p for p in getattr(op, "SCOPE_NEEDS", ()) if p not in driven]
            if miss:
                why.append(f"{type(op).__name__} restarts at a frame start only when a control drives {miss}")
    return ("stream", why) if why else ("frame", [])

def plan(marker, steps, controls, where):
    """The Region for `marker` (a step id, or None for no region), checked as above; `controls` is {header id: Entry}"""
    if marker is None:
        return None
    ids = [s.id for s in steps]
    if ids.count(marker) != 1:
        raise ConfigError(f"{where}: parallel starts at {marker!r}: "
                          f"{'no such step' if marker not in ids else 'several steps have that id'} (step ids: {ids})")
    first = ids.index(marker)
    region = steps[first:]
    in_region = {s.id for s in region}
    driven = {}
    for header, entry in controls.items():
        for sid, params in entry.drive.items():
            if sid in in_region:
                driven.setdefault(sid, set()).update(params)
        if header in in_region:
            raise ConfigError(f"{where}: parallel from {marker!r}, but {header!r} publishes fields and is inside it: a "
                              f"header sits in the serial front, the region takes the fields it resolved")
        for sid in entry.drive:
            if sid in ids and ids.index(sid) < ids.index(header) and ids.index(sid) >= first:
                raise ConfigError(f"{where}: {header!r} drives {sid!r} after the fact (a late change, which the serial "
                                  f"runner re-runs): not allowed in a parallel region")
    for st in region:
        s, why = scope_of(st, driven.get(st.id, ()))
        if s != FRAME:
            raise ConfigError(f"{where}: step {st.id!r} ({st.category}:{st.kind}) is in the parallel region from {marker!r} "
                              f"but is not frame scoped: {'; '.join(why)}")
    return Region(first, region, {k: sorted(v) for k, v in driven.items()})

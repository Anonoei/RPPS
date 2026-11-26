"""Profiles: one config file per signal, both directions (rpps.profile)

    {"base": "profile", "comment": "...", "ref": "...",
     "params": {"sps": {"default": 4}},
     "defaults": {"timing": {"method": "mat"}},           optional: how the receiver estimates what sync steps name
     "steps": [ {"<category>": "<kind>", "id": ..., ...}, ... ]}     in receive order (see rpps.profile.steps)
     "controls": {"<header step id>": {...}}              optional: header-driven control (see rpps.profile.control)
     "parallel": "<step id>"                              optional: the first step that runs frame by frame (see rpps.profile.parallel)
"""
import copy

from .. import _config
from ..base.types import Rep
from ..errors import ConfigError
from . import steps as _steps
from . import control as _control
from . import parallel as _parallel
from .runner import Runner, Slot

def _warn(msg):
    import warnings
    from ..errors import ConfigWarning
    warnings.warn(msg, ConfigWarning, stacklevel=4)

def _tracked(obj):
    """The profile and each of its steps (a branch's too) as rpps._config.Keys, which record what building reads"""
    def step(sp):
        if not isinstance(sp, dict):
            return sp
        sp = _config.Keys(dict.items(sp))
        if "branches" in dict.keys(sp) and isinstance(dict.get(sp, "branches"), dict):
            dict.__setitem__(sp, "branches", {k: [step(x) for x in v] for k, v in dict.items(dict.get(sp, "branches"))})
        return sp
    out = _config.Keys(dict.items(obj))
    dict.__setitem__(out, "steps", [step(x) for x in dict.get(obj, "steps")])
    return out

class Profile:
    """A signal: its steps in receive order, built from a definition. `rx()` and `tx()` give runners"""
    def __init__(self, name, obj, sync=None, params=None):
        self.name, self.obj = name, obj
        self.comment, self.ref = obj.get("comment", ""), obj.get("ref", "")
        self.params = dict(params or {})
        self._sync = {k: _config.Keys(dict(v)) for k, v in (sync or {}).items()} # (what reads them is recorded: _check_keys)
        if "steps" not in obj or not obj["steps"]:
            raise ConfigError(f"profile {name}: no steps")
        self.obj = obj = _tracked(obj)
        self.steps = self._build()
        branch = [self.steps[-1]] if self.steps and self.steps[-1].category == "branch" else [] # (its `frame` can be driven)
        self.controls = _control.parse(obj.get("controls"), self.flat(self.steps) + branch, f"profile {name}")
        if self.steps[-1].category == "branch" and obj.get("parallel"):
            raise ConfigError(f"profile {name}: a parallel region and branches together are not supported")
        self.region = _parallel.plan(obj.get("parallel"), self.steps, self.controls, f"profile {name}")
        self.capture = self._captures(obj.get("capture"))
        self._check_keys()

    def _captures(self, spec):
        """The capture rules ("capture": rpps.link.capture): one per message kind the profile gives"""
        from ..link.capture import Captures
        spec = dict(spec or {})
        unknown = set(spec) - set(self.kinds()) - set(self.kinds(parallel=True))
        if unknown:
            raise ConfigError(f"profile {self.name}: capture rules for kinds it doesn't give: {sorted(unknown)} "
                              f"(gives {sorted(self.kinds())})")
        return Captures(spec, f"profile {self.name}: capture", self.params)

    # A profile's keys others read: what it needs to run (rpps.profile.info, the catalogue)
    READ_ELSEWHERE = ("requires",)
    # A step's keys the runner reads when it runs (its message; a branch step's framing), not when it is built
    LATER = ("message",)
    LATER_BRANCH = ("frame", "take", "select")

    def _check_keys(self):
        """A ConfigWarning for each key of the profile, its steps, its sync defaults and the load-time `sync` nothing read"""
        where = f"profile {self.name}"
        _config.warn_unused(self.obj, where, _config.DEFINITION_KEYS + self.READ_ELSEWHERE, stacklevel=4)
        def specs(items):
            for sp in items:
                yield sp
                if isinstance(sp, dict) and "branch" in dict.keys(sp):
                    for chain in dict.get(sp, "branches", {}).values():
                        yield from specs(chain)
        for sp in specs(dict.get(self.obj, "steps", [])):
            if not isinstance(sp, _config.Keys):
                continue
            cat = next((k for k in dict.keys(sp) if k in _steps.REGISTRY or k == "branch"), None)
            sid = dict.get(sp, "id", dict.get(sp, cat) if cat else "?")
            if cat == "sync" and "method" in dict.keys(self._sync.get(dict.get(sp, "sync"), {})):
                continue # (its method replaced at load: the step's own settings were for the one it names)
            exempt = (cat, "id", "comment") + self.LATER + (self.LATER_BRANCH + ("branches", "order") if cat == "branch" else ())
            _config.warn_unused(sp, f"{where}: step {sid!r}", exempt, stacklevel=4)
        reads = getattr(self, "_ctx", {}).get("method_reads", {})
        for label, given in (("defaults", dict.get(self.obj, "defaults", None) or {}), ("sync", self._sync)):
            for kind, settings in given.items():
                if not isinstance(settings, dict):
                    continue
                seen = set().union(*reads.get(kind, [set()]), getattr(settings, "read", ()))
                left = sorted(k for k in settings if k not in seen and k != "comment")
                if left and kind not in reads:
                    _warn(f"{where}: {label} {kind!r}: no {kind} step takes {left}")
                elif left:
                    _warn(f"{where}: {label} {kind!r}: {', '.join(repr(k) for k in left)} not used by its method")

    @staticmethod
    def flat(steps):
        """Every step: the trunk's, then each branch's (a branch step holds its branches' steps)"""
        if steps and steps[-1].category == "branch":
            return steps[:-1] + [st for chain in steps[-1].obj.values() for st in chain]
        return list(steps)

    def _build(self):
        """The steps, built (each runner builds its own: header-driven control changes their operators)"""
        ctx = _steps.context(self.params, self.obj.get("defaults", {}), self._sync)
        self._ctx = ctx
        specs = list(self.obj["steps"])
        at = [i for i, s in enumerate(specs) if "branch" in s]
        if at and (len(at) > 1 or at[0] != len(specs) - 1):
            raise ConfigError(f"profile {self.name}: one branch step, the last (it ends the chain)")
        built = self._chain(specs[:-1] if at else specs, Rep.SAMPLES, ctx)
        if at:
            built.append(self._branch(specs[-1], built[-1].gives if built else Rep.SAMPLES, ctx))
        return built

    def _branch(self, spec, rep, ctx):
        """The branch step: each branch's steps built as a chain from the trunk's output (rpps.profile.branch)"""
        from .steps import Step
        where = f"profile {self.name}: branch {spec['branch']!r}"
        branches = spec.get("branches")
        if not isinstance(branches, dict) or not branches:
            raise ConfigError(f"{where}: needs 'branches': {{name: [steps]}}")
        order = list(spec.get("order", branches))
        if sorted(order) != sorted(branches):
            raise ConfigError(f"{where}: order {order} is not the branches {sorted(branches)}")
        nested = {name: self._chain(list(branches[name]), rep, ctx) for name in order}
        ids = [st.id for chain in nested.values() for st in chain]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            raise ConfigError(f"{where}: step ids {dup} are in several branches (a control names one step)")
        return Step("branch", str(spec["branch"]), [], [], rep, rep, spec, nested, str(spec.get("id", spec["branch"])),
                    {"frame": "live"} if "frame" not in spec else {}) # (frames a control starts, when it has no fixed length)

    def _chain(self, specs, rep, ctx):
        """A list of step specs built in order from `rep`"""
        # A step can refer to one after it (a pilot equalizer to its carriers): build those first
        early = {i: _steps.build(s, None, ctx) for i, s in enumerate(specs) if s.get("symbol") == "carriers"}
        built = []
        for i, spec in enumerate(specs):
            st = early.get(i) or _steps.build(spec, rep, ctx)
            st.takes = rep
            rep = st.gives
            built.append(st)
        if self._sync.get("timing", {}).get("fuse", True):
            self._fuse(built)
        for i, st in enumerate(built):
            if st.category == "sync" and st.kind == "preamble" and ("sequence" in st.spec or "word" in st.spec):
                self._template(i, specs, built)
            elif st.category == "sync" and st.kind == "frame" and getattr(st.obj, "preamble", 0) is None:
                self._frame_template(i, specs, built)
        return built

    def _frame_template(self, i, specs, built):
        """A frame sync's preambles: words given as bits through the transmit side (`_symbols`), complex ones as they are"""
        words = _steps.frame_words(built[i].spec, f"profile {self.name}: frame {built[i].id!r}")
        made = {n: (self._symbols(i, specs, built, v) if kind == "bits" else v, f) for n, kind, v, f in words}
        built[i].obj.set_template(made[None][0] if None in made else made)

    def _symbols(self, i, specs, built, bits):
        """Bits as the steps after step i make them on transmit, down to the symbols it takes (a frame sync's words)"""
        import numpy as np
        from ..base.data import Data
        from .steps import STAGE, ENCODE
        where = f"profile {self.name}: {built[i].kind} {built[i].id!r}"
        j = next((k for k in range(i + 1, len(built)) if built[k].gives == Rep.CHANNEL), None)
        if j is None:
            raise ConfigError(f"{where}: no step after it maps symbols to bits, to make its symbols with")
        ctx = _steps.context(self.params, self.obj.get("defaults", {}), self._sync)
        fresh, rep = [], built[i].gives
        for k in range(i + 1, j + 1):
            st = _steps.build(specs[k], rep, ctx)
            rep = st.gives
            fresh.append(st)
        d = Data(np.asarray(bits, bool), flush=True, sps=1)
        d = d.derive(d.data, d.CAT, d.DT, rep=Rep.CHANNEL)
        for st in reversed(fresh):
            for how, op in st.tx:
                d = d + op if how == STAGE else d * op if how == ENCODE else d / op
        return np.asarray(d.data)

    def _template(self, i, specs, built):
        """A preamble (or frame) step's waveform: its bits through the transmit side of the steps after it, from the first
        that makes channel bits back to the step's input (built anew: making it changes no operator the profile uses). Its
        first symbol's peak is where the group delay of those steps puts it"""
        import numpy as np
        from ..base.data import Data
        spec, where = built[i].spec, f"profile {self.name}: {built[i].kind} {built[i].id!r}"
        j = next((k for k in range(i + 1, len(built)) if built[k].gives == Rep.CHANNEL), None)
        if j is None:
            raise ConfigError(f"{where}: no step after it maps symbols to bits, to make its waveform with")
        if "sequence" in spec:
            from .. import sequence
            bits = np.asarray(sequence.load(spec["sequence"]).bits, bool)
        else:
            from ..sequence.word import word_bits
            bits = np.asarray(word_bits(spec["word"], spec["bits"], spec.get("order", "msb")), bool)
        bits = np.concatenate((np.asarray(spec.get("lead", []), bool), bits))
        ctx = _steps.context(self.params, self.obj.get("defaults", {}), self._sync)
        fresh, rep = [], built[i].gives
        for k in range(i + 1, j + 1):
            st = _steps.build(specs[k], rep, ctx)
            rep = st.gives
            fresh.append(st)
        d = Data(bits, flush=True, sps=1) # (one sample a symbol: the transmit side's shaping takes it up to the stream's)
        d = d.derive(d.data, d.CAT, d.DT, rep=Rep.CHANNEL)
        from .steps import STAGE, ENCODE
        for st in reversed(fresh):
            for how, op in st.tx:
                d = d + op if how == STAGE else d * op if how == ENCODE else d / op
        x = np.asarray(d.data)
        built[i].obj.set_template(x, int(round(d.meta.delay*d.meta.Fs)), int(round(len(bits)*(d.meta.sps or 1))))

    @staticmethod
    def _fuse(steps):
        """A matched filter right before a timing that evaluates it at the instants it samples (Known, MatTrack) is taken over by
        it: the step of the filter keeps no operator (Oerder & Meyr needs the filter's output at every sample, so nothing fuses)"""
        from ..sync.time import Known, MatTrack
        from ..wave import Matched
        for a, b in zip(steps, steps[1:]):
            if (a.category == "wave" and a.kind == "shape" and len(a.rx) == 1 and isinstance(a.rx[0][1], Matched) and a.rx[0][1].unit
                    and b.category == "sync" and b.kind == "timing" and len(b.rx) == 1 and isinstance(b.rx[0][1], (Known, MatTrack))):
                b.rx[0][1].fuse(a.rx[0][1])
                a.rx = []

    # ---- message kinds

    @staticmethod
    def _emits(st):
        """The kind of message a step gives (rpps.profile.message), None for none: its `"message"` cut's kind, else its id
        when an operator of it emits (`EMITS`)"""
        if st.spec.get("message"):
            return st.spec["message"].get("kind") or st.id
        if any(getattr(op, "EMITS", False) for _, op in st.rx):
            return st.id
        return None

    def kinds(self, parallel=False):
        """{kind: step id} of the messages the receive side gives. parallel: as run with its parallel region, whose kernels
        give one `"frame"` message a frame in place of what the region's steps would emit"""
        region = {st.id for st in self.region.steps} if parallel and self.region is not None else set()
        out = {}
        for st in self.flat(self.steps):
            k = self._emits(st)
            if k is not None and st.id not in region:
                out[k] = st.id
        if region:
            out["frame"] = self.region.steps[-1].id
        return out

    def needed(self, kinds, parallel=False):
        """The step ids the messages of `kinds` need: their emitters, and every header whose fields drive a needed step (and
        so on). Raises ConfigError for a kind the profile doesn't give"""
        have = self.kinds(parallel)
        kinds = set(kinds)
        unknown = kinds - set(have)
        if unknown:
            raise ConfigError(f"profile {self.name}: no message kinds {sorted(unknown)} (gives {sorted(have)})")
        need = {have[k] for k in kinds}
        grew = True
        while grew:
            grew = False
            for e in self.controls.values():
                if e.header not in need and any(sid in need for sid in e.drive):
                    need.add(e.header)
                    grew = True
        return need

    def _prune(self, steps, need):
        """steps, without what the needed steps don't need: a chain ends at its last needed step (those before it feed it),
        a branch step keeps the branches that have one (it goes when none has)"""
        if steps and steps[-1].category == "branch":
            br = steps[-1]
            nested = {}
            for name, chain in br.obj.items():
                chain = self._cut(chain, need)
                if chain:
                    nested[name] = chain
            if nested:
                import copy
                br = copy.copy(br)
                br.obj = nested
                return list(steps[:-1]) + [br]
            steps = steps[:-1]
        return self._cut(steps, need)

    @staticmethod
    def _cut(chain, need):
        last = max((i for i, st in enumerate(chain) if st.id in need), default=-1)
        return list(chain[:last + 1])

    # ---- runners

    def rx(self, start=Rep.SAMPLES, applied=(), parallel=None, source=None, kinds=None):
        """Receive runner. start: the representation the source gives (a demodulator's timed symbols are SYMBOLS): the steps
        before it are skipped. applied: sync kinds the source has done already (`{"timing"}`): skipped too.
        source: an rp.source.Source, which says both (its REP and APPLIED).
        parallel: an `rp.parallel.Parallel` (where and how the profile's parallel region runs, frame by frame); the profile
        must have one (`"parallel"` in its definition)
        kinds: the message kinds wanted (`kinds()`), None for all: only those come out of `take_messages()`, and only the
        steps they need run (`needed`): branches none of them needs are not built, a chain ends at its last needed step
        (the runner's output is that step's). A parallel region runs whole or not at all: not when none of it is needed"""
        if source is not None:
            start, applied = source.REP, source.APPLIED
        if parallel is not None and self.region is None:
            raise ConfigError(f"profile {self.name}: has no parallel region (no \"parallel\" in its definition)")
        keep = lambda st: st.category == "branch" or not (st.takes.value < start.value or (st.category == "sync" and st.kind in applied))
        if kinds is None:
            prune = lambda steps: steps
        else:
            kinds = frozenset(kinds)
            need = self.needed(kinds, parallel is not None)
            if parallel is not None:
                if need & {st.id for st in self.region.steps}:
                    need |= {st.id for st in self.region.steps}
                else:
                    parallel = None # (nothing of the region is needed: it isn't run)
            prune = lambda steps: self._prune(steps, need)
        steps = prune([st for st in self._build() if keep(st)])
        if not steps:
            raise ConfigError(f"profile {self.name}: no step left to receive {sorted(kinds)} with")
        if steps[-1].category == "branch":
            run = self._branch_rx(steps, parallel)
        elif (k := next((i for i, st in enumerate(steps) if st.category == "sync" and st.kind == "burst"), None)) is not None:
            run = self._burst_rx(steps, k, lambda: prune([st for st in self._build() if keep(st)]), parallel)
        else:
            run = self._chain_rx(steps, parallel)
        run.kinds = kinds
        return run

    def _chain_rx(self, steps, parallel):
        slots = self._slots(steps)
        if parallel is None:
            return Runner(slots, f"{self.name} rx", self.controls, "rx", profile=self.name)
        from ..parallel.runner import ParallelRunner
        first = self.region.steps[0].id
        from ..parallel.worker import RegionSteps
        make = RegionSteps(self.name, self.obj, self._sync, self.params, self.region.first)
        return ParallelRunner(slots, f"{self.name} rx (parallel)", self.controls, "rx", make, first, parallel, profile=self.name)

    @staticmethod
    def _slots(steps):
        slots = []
        for st in steps:
            slots += [Slot(how, op, st, len(slots) + i) for i, (how, op) in enumerate(st.rx)]
        return slots

    def _burst_rx(self, steps, k, build, parallel):
        """The receive side of a profile with a burst step (rpps.profile.burst): the steps after it built anew per burst"""
        from .burst import BurstRunner
        where = f"profile {self.name}"
        if parallel is not None:
            raise ConfigError(f"{where}: a burst step and a parallel region together are not supported")
        if sum(st.kind == "burst" and st.category == "sync" for st in steps) > 1:
            raise ConfigError(f"{where}: one burst step")
        pre = Runner(self._slots(steps[:k]), f"{self.name} rx", self.controls, "rx", profile=self.name) if k else None
        made = []
        def make_post():
            """The steps after the burst step, a copy of them built once: a burst's controls change its own operators,
            and copying is many times cheaper than building from the definitions (a burst can be a millisecond)"""
            if not made:
                made.append(build()[k + 1:])
            post = copy.deepcopy(made[0])
            return Runner(self._slots(post), f"{self.name} rx (burst)", self.controls, "rx", profile=self.name)
        return BurstRunner(pre, steps[k].obj, make_post, f"{self.name} rx", self.controls.get(steps[k].id))

    def _branch_rx(self, steps, parallel):
        """The receive side of a profile that ends in branches (rpps.profile.branch)"""
        import copy
        from .branch import BranchRunner, BranchChain, _Bus, Exit, EXIT
        from .steps import Step, STAGE
        if parallel is not None:
            raise ConfigError(f"profile {self.name}: branches and a parallel region together are not supported")
        if any(st.category == "sync" and st.kind == "burst" for st in self.flat(steps)):
            raise ConfigError(f"profile {self.name}: a burst step and branches together are not supported")
        nested = steps[-1].obj
        owner = {st.id: name for name, chain in nested.items() for st in chain}
        for e in self.controls.values():
            frm = owner.get(e.header)
            for sid in e.drive:
                to = owner.get(sid)
                if frm is not None and to is not None and list(nested).index(to) < list(nested).index(frm):
                    raise ConfigError(f"profile {self.name}: control {e.header!r} in branch {frm!r} drives {sid!r} in {to!r}, "
                                      f"which runs before it (order {list(nested)})")
        bus = _Bus(self.controls, owner)
        trunk_ids = {st.id for st in steps[:-1]}
        entries, exit_params = dict(self.controls), []
        bid = steps[-1].id
        framed = any(bid in e.drive for e in self.controls.values())
        for h, e in self.controls.items(): # A trunk header driving branch steps drives the trunk's exit in their place
            outside = {sid: ps for sid, ps in e.drive.items() if sid in owner or sid == bid} # (the branch step: frame starts)
            if h in trunk_ids and outside:
                e2 = copy.copy(e)
                e2.drive = {sid: ps for sid, ps in e.drive.items() if sid not in owner}
                e2.drive[EXIT] = {f"{sid}/{p}": src for sid, ps in outside.items() for p, src in ps.items()}
                exit_params += list(e2.drive[EXIT])
                entries[h] = e2
        trunk_steps, exit_op = list(steps[:-1]), None
        if exit_params:
            exit_op = Exit(exit_params)
            trunk_steps.append(Step("signal", "exit", [(STAGE, exit_op)], [], steps[-1].takes, steps[-1].takes, {}, exit_op, EXIT,
                                    dict(exit_op.CONTROLS)))
        trunk = Runner(self._slots(trunk_steps), f"{self.name} rx", entries, "rx", profile=self.name) if trunk_steps else None
        chains = {name: BranchChain(self._slots(chain), f"{self.name} rx {name}", self.controls, name, bus, profile=self.name)
                  for name, chain in nested.items()}
        spec = steps[-1].spec
        return BranchRunner(trunk, chains, f"{self.name} rx", bus, list(nested), spec.get("frame"), exit_op, owner,
                            spec.get("take"), spec.get("select"), bid, framed)

    def tx(self):
        """Transmit runner: the transform steps inverted, in reverse order (most sync and signal steps have no transmit side).
        Call it as `runner(block, fields=...)` to set the fields of the frames that start in the block (see Runner).
        A profile with a burst step gives a burst transmitter (rpps.profile.burst.BurstTxRunner): a call a burst"""
        built = self._build()
        if built[-1].category == "branch":
            raise ConfigError(f"profile {self.name}: transmitting through branches is not built (receive only)")
        k = [i for i, st in enumerate(built) if st.category == "sync" and st.kind == "burst"]
        if k:
            return self._burst_tx(built, k)
        return Runner(self._tx_slots(built), f"{self.name} tx", self.controls, "tx", profile=self.name)

    @staticmethod
    def _tx_slots(steps):
        slots = []
        for st in reversed(steps):
            slots += [Slot(how, op, st, len(slots) + i) for i, (how, op) in enumerate(st.tx)]
        return slots

    def _burst_tx(self, built, k):
        """The transmit side of a profile with a burst step: the steps after it built anew for every burst, the bursts
        placed on one stream by the step's rpps.sync.BurstTx, the steps before it on that stream"""
        from .burst import BurstTxRunner
        if len(k) > 1:
            raise ConfigError(f"profile {self.name}: one burst step")
        k = k[0]
        pre = Runner(self._tx_slots(built[:k]), f"{self.name} tx", self.controls, "tx", profile=self.name) if k else None
        def make_post():
            post = self._build()[k + 1:]
            return Runner(self._tx_slots(post), f"{self.name} tx (burst)", self.controls, "tx", profile=self.name)
        return BurstTxRunner(make_post, built[k].tx[0][1], pre, f"{self.name} tx", self.controls.get(built[k].id))

    def __repr__(self):
        return f"<Profile {self.name}: " + ", ".join(f"{s.category}:{s.kind}" for s in self.steps) + ">"

    @staticmethod
    def load(name, obj):
        """Loader of the profile domain (`_config.load_obj`): the parameters are already resolved into the steps"""
        return Profile(name, obj, params=_config.declared(obj))

def load(name, sync=None, **params):
    """Load a profile: `rp.profile.load("ieee802154/2450", sps=8)`. params: values for its "params".
    sync: per sync kind, the method and its settings, over the profile's defaults: `sync={"timing": {"method": "known"}}`"""
    file = _config.find("profile", name)
    if file is None:
        raise FileNotFoundError(f"profile/{name}: not found, have {_config.ls('profile')}")
    resolved = _config.resolve(_config.read(file), params, f"profile/{name}")
    return Profile(name, resolved, sync=sync, params=_config.declared(resolved))

def ls(blind=False):
    """The profiles there are: {family: [modes]}

    blind: only what the blind tools try (rp._config.ls): no definitions tagged "example", and only the selected tags (rp.select_tags)"""
    return _config.ls("profile", blind=blind)

DOMAIN = _config.register_domain("profile", {"profile": Profile.load})

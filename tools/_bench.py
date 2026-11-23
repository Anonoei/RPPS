"""Results of the tools/ benchmarks, kept and compared per machine in .benchmarks/ (gitignored: timings and thresholds
are of this computer and checkout, not of the code alone)

A benchmark adds the flags with `add_args(ap)`, records each number with `Record.add`, and ends with
`return finish(args, rec)`:

    python tools/<bench>.py --save              # also keep this run: .benchmarks/<bench>/runs/<commit>-<time>.json
    python tools/<bench>.py --update-baseline   # this run is the baseline: .benchmarks/<bench>/baseline.json
    python tools/<bench>.py --check             # compare with the baseline; exit 1 on a regression
    python tools/<bench>.py --compare FILE      # compare with any saved run

A result is worse when it moves the wrong way by more than its tolerance: relative (--threshold, default 25%) for
rates and times, absolute (in its unit) where `add` gives `tol` (an Es/N0 threshold in dB). A baseline from another
machine, another native-kernel state or other options is compared with a warning.
"""
import json
import math
import platform
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / ".benchmarks"
OWN = ("save", "update_baseline", "check", "compare", "threshold")

def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""

def _cpu():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()

def meta(args=None):
    """Where a run is from: checkout, machine, versions, native kernels, the benchmark's own options"""
    import numpy as np
    try:
        from rpps import _accel
        native = bool(_accel.available())
    except ImportError:
        native = None
    opts = {k: v for k, v in vars(args).items() if k not in OWN} if args is not None else {}
    return {"commit": _git("rev-parse", "--short", "HEAD") or "unknown",
            "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "cpu": _cpu(), "machine": platform.machine(),
            "python": platform.python_version(), "numpy": np.__version__, "native": native,
            "args": json.loads(json.dumps(opts, default=str))}

class Record:
    """A benchmark's results, by name: value, unit, which way is better, tolerance"""
    def __init__(self, bench):
        self.bench = bench
        self.results = {}

    def add(self, name, value, unit, better="higher", tol=None):
        """value None: nothing measured (never reached, not run). tol: an absolute tolerance in `unit`; None: relative"""
        assert better in ("higher", "lower"), better
        v = None if value is None else float(value)
        self.results[name] = {"value": v, "unit": unit, "better": better, "tol": tol}
        return value

def add_args(ap):
    g = ap.add_argument_group("results (.benchmarks/, per machine)")
    g.add_argument("--save", action="store_true", help="also keep this run in .benchmarks/<bench>/runs/")
    g.add_argument("--update-baseline", action="store_true", help="write this run as the baseline")
    g.add_argument("--check", action="store_true", help="compare with the baseline; exit 1 on a regression")
    g.add_argument("--compare", metavar="PATH", help="compare with a saved run")
    g.add_argument("--threshold", type=float, default=0.25, help="relative change that counts (rates, times)")
    return ap

def _save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1))

def compare(base, cur, threshold):
    """[(name, status, base value, value, change)], status regression / improvement / ok / new / gone"""
    rows = []
    for name, c in cur["results"].items():
        b = base["results"].get(name)
        if b is None:
            rows.append((name, "new", None, c["value"], None))
            continue
        bv, cv = b["value"], c["value"]
        if bv is None or cv is None:
            status = "ok" if bv is cv else ("regression" if cv is None else "improvement")
            rows.append((name, status, bv, cv, None))
            continue
        sign = 1 if c["better"] == "higher" else -1
        if c["tol"] is not None:
            change = cv - bv
            gain, tol = sign*change, c["tol"]
        else:
            change = (cv - bv)/abs(bv) if bv else (0.0 if cv == bv else math.copysign(math.inf, cv - bv))
            gain, tol = sign*change, threshold
        status = "regression" if gain < -tol else "improvement" if gain > tol else "ok"
        rows.append((name, status, bv, cv, change))
    rows += [(name, "gone", b["value"], None, None) for name, b in base["results"].items() if name not in cur["results"]]
    return rows

def _fmt(v, unit):
    return "-" if v is None else f"{v:.4g} {unit}"

def finish(args, rec):
    """Save / baseline / compare as the flags say. Returns the exit code (1: --check found a regression)"""
    cur = {"bench": rec.bench, "meta": meta(args), "results": rec.results}
    m = cur["meta"]
    where = DIR / rec.bench
    if args.save:
        tag = f"{m['commit']}{'-dirty' if m['dirty'] else ''}-{m['time'].replace(':', '')}"
        _save(where / "runs" / f"{tag}.json", cur)
        print(f"saved: {where / 'runs' / f'{tag}.json'}")
    if args.update_baseline:
        _save(where / "baseline.json", cur)
        print(f"baseline written: {where / 'baseline.json'}")
        return 0
    path = Path(args.compare) if args.compare else where / "baseline.json" if args.check else None
    if path is None:
        return 0
    if not path.exists():
        if args.check:
            _save(path, cur)
            print(f"no baseline yet; this run is now {path}")
            return 0
        print(f"no such file: {path}")
        return 2
    base = json.loads(path.read_text())
    bm = base["meta"]
    if (bm.get("cpu"), bm.get("machine")) != (m["cpu"], m["machine"]):
        print(f"WARNING: the baseline is from another machine ({bm.get('cpu')}): the comparison is unreliable")
    if bm.get("native") != m["native"]:
        print(f"WARNING: native kernels {bm.get('native')} in the baseline, {m['native']} now")
    diff = {k for k in set(bm.get("args", {})) | set(m["args"]) if bm.get("args", {}).get(k) != m["args"].get(k)}
    if diff:
        print(f"WARNING: other options than the baseline's: {', '.join(sorted(diff))}")
    rows = compare(base, cur, args.threshold)
    print(f"\ncompared with {bm.get('commit')}{'+dirty' if bm.get('dirty') else ''} ({bm.get('time')}):")
    mark = {"regression": "WORSE", "improvement": "better", "new": "new", "gone": "gone", "ok": ""}
    for name, status, bv, cv, change in rows:
        if status == "ok" and (change is None or change == 0):
            continue
        r = cur["results"].get(name) or base["results"][name]
        ch = "" if change is None else f"  ({change:+.2f} {r['unit']})" if r["tol"] is not None else f"  ({change:+.0%})"
        print(f"  {mark[status]:<7} {name:<44} {_fmt(bv, r['unit'])} -> {_fmt(cv, r['unit'])}{ch}")
    bad = [r for r in rows if r[1] == "regression"]
    print(f"{len(rows)} compared, {len(bad)} worse, {sum(r[1] == 'improvement' for r in rows)} better")
    return 1 if bad and args.check else 0

if __name__ == "__main__":
    print(__doc__)

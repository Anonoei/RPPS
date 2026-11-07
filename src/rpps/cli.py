"""
RPPS command line interface

    python -m rpps ls [category ...]
    python -m rpps survey <file> --fs <Hz> [--fmt <format>] [--cf <Hz>] [--plot]
    python -m rpps fmt [--check] [dir]
    python -m rpps gen-ldpc <dvb_s2_tables.hh> [table ids ...] [--out dir]
"""
import argparse
import json
import re
import shutil
import sys
import textwrap
from pathlib import Path


def _categories(rp):
    """name -> (title, callable returning {group: [names]})"""
    return {
        "mod": ('Modulation  (rp.mod.load("<name>"[, "<mapping>"]))', rp.mod.ls),
        "coding": ('Coding  (rp.coding.load(<kind>, "<family>/<name>"))', rp.coding.ls),
        "scram": ('Scrambling  (rp.scram.load(<kind>, "<name>"))', rp.scram.ls),
        "encap": ('Encapsulation  (rp.encap.load(<kind>, "<name>"))', rp.encap.ls),
        "chain": ('Chains  (rp.Chain.load("<standard>/<name>"))', lambda: rp._config.ls("chain")),
        "filters": ('Filters  (rp.filters.load(<kind>, "<family>/<name>"))', rp.filters.ls),
        "windows": ("Built-in Windows  (rp.filters.Window.get(<name>))", lambda: {"windows": rp.filters.Window.s()}),
    }


def _show(title, groups, width):
    """Print a title, then one wrapped, aligned line per group"""
    print(f"\n{title}")
    if not groups:
        print("  (none)")
        return
    pad = max(len(k) for k in groups)
    for k, names in groups.items():
        lead = f"  {k:<{pad}}  "
        print(textwrap.fill(", ".join(names), width - len(lead), initial_indent=lead, subsequent_indent=" " * len(lead)))


def cmd_ls(args):
    import rpps as rp
    cats = _categories(rp)
    width = min(shutil.get_terminal_size((100, 20)).columns, 120)
    for name in args.category or cats:
        title, fn = cats[name]
        _show(title, fn(), width)
    return 0


def cmd_fmt(args):
    """Rewrite config/**/*.json in the house style; with --check only list the files that would change"""
    from rpps._config import PATH_CONFIG, format_json
    root = Path(args.dir) if args.dir else PATH_CONFIG
    changed = []
    for f in sorted(root.rglob("*.json")):
        obj = json.loads(f.read_text())
        text = format_json(obj) + "\n"
        assert json.loads(text) == obj, f
        if text != f.read_text():
            changed.append(f)
            if not args.check:
                f.write_text(text)
    for f in changed:
        print(f.relative_to(root))
    return 1 if changed and args.check else 0


def cmd_survey(args):
    """Survey a capture and print each signal found; --plot also shows the spectrum (needs matplotlib)"""
    import rpps as rp
    src = rp.serial.File(args.fmt, args.file, Fs=args.fs, cf=args.cf)
    sv = rp.dsp.survey(src, samples=args.samples)
    print(f"{src.path}: {sv.Fs/1e6:g} MS/s at {sv.cf/1e6:g} MHz, {len(sv)} signals, {sv.bin_hz:.0f} Hz bins, floor {sv.floor:.1f} dB")
    for i, s in enumerate(sv):
        print(f"  {i:>2}  {s.describe()}")
    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("--plot needs matplotlib (pip install rpps[viz])", file=sys.stderr)
            return 1
        sv.plot()
        plt.show()
    return 0


_LDPC_RATES = {"B1": "1_4", "B2": "1_3", "B3": "2_5", "B4": "1_2", "B5": "3_5", "B6": "2_3", "B7": "3_4", "B8": "4_5",
               "B9": "5_6", "B10": "8_9", "B11": "9_10"}


def _ldpc_parse(text):
    out = {}
    for m in re.finditer(r"struct DVB_S2_TABLE_(\w+)\s*\{(.*?)\n\};", text, re.S):
        body = m.group(2)

        def arr(name):
            a = re.search(name + r"\[\]\s*=\s*\{(.*?)\};", body, re.S)
            return [int(x) for x in re.findall(r"\d+", a.group(1))] if a else None

        def const(n):
            return int(re.search(rf"static const int {n} = (\d+);", body).group(1))

        out[m.group(1)] = dict(M=const("M"), N=const("N"), K=const("K"), DEG=arr("DEG"), LEN=arr("LEN"), POS=arr("POS"))
    return out


def _ldpc_rows(t):
    pos, out = iter(t["POS"]), []
    for deg, n in zip(t["DEG"], t["LEN"]):
        if deg == 0:
            break
        for _ in range(n):
            out.append([next(pos) for _ in range(deg)])
    return out


def cmd_gen_ldpc(args):
    """Turn DVB-S2 LDPC parity address tables (EN 302 307 annex B/C) into config/coding/block/ldpc files

    The input is xdsopl/LDPC's dvb_s2_tables.hh (https://github.com/xdsopl/LDPC). It is read as data, never executed.
    """
    from rpps._config import PATH_CONFIG
    tables = _ldpc_parse(Path(args.tables).read_text())
    root = Path(args.out) if args.out else PATH_CONFIG / "coding" / "block" / "ldpc"
    root.mkdir(parents=True, exist_ok=True)
    for i in args.ids or ["B5", "B7"]:
        t = tables[i]
        r = _ldpc_rows(t)
        assert len(r) * t["M"] == t["K"], (i, len(r), t["K"])
        name = f"dvbs2_64800_{_LDPC_RATES[i]}"
        (root / f"{name}.json").write_text(json.dumps({
            "base": "block", "type": "ldpc", "n": t["N"], "k": t["K"], "group": t["M"], "systematic": "head", "iters": 50, "alpha": 0.8,
            "comment": f"DVB-S2 normal frame LDPC rate {_LDPC_RATES[i].replace('_', '/')}: parity address table B.{i[1:]} of EN 302 307",
            "table": r}, separators=(",", ":")))
        print(name, "rows", len(r), "k", t["K"])
    return 0


def main(argv=None):
    # Category names are static so --help doesn't need to import rpps
    names = ["mod", "coding", "scram", "encap", "chain", "filters", "windows"]
    parser = argparse.ArgumentParser(prog="rpps", description="RF Parameter Processor Suite")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_ls = sub.add_parser("ls", help="list available modulations, codings, chains, filters, ...")
    p_ls.add_argument("category", nargs="*", choices=names, help="limit output to these categories (default: all)")
    p_ls.set_defaults(fn=cmd_ls)

    p_sv = sub.add_parser("survey", help="find and list the signals in a capture file")
    p_sv.add_argument("file", help="capture file")
    p_sv.add_argument("--fs", type=float, required=True, help="sample rate [Hz]")
    p_sv.add_argument("--fmt", default="cf64", help="sample format, see rp.serial.formats.ls() (default: cf64, complex float32 pairs)")
    p_sv.add_argument("--cf", type=float, default=0.0, help="centre frequency [Hz] (default: 0)")
    p_sv.add_argument("--samples", type=int, default=2**17, help="samples to read (default: 131072)")
    p_sv.add_argument("--plot", action="store_true", help="show the survey plot (needs matplotlib)")
    p_sv.set_defaults(fn=cmd_survey)

    p_fmt = sub.add_parser("fmt", help="rewrite config json files in one style")
    p_fmt.add_argument("dir", nargs="?", help="config directory (default: the shipped config)")
    p_fmt.add_argument("--check", action="store_true", help="only list files that would change, exit 1 if any")
    p_fmt.set_defaults(fn=cmd_fmt)

    p_gen = sub.add_parser("gen-ldpc", help="generate DVB-S2 LDPC config files from dvb_s2_tables.hh")
    p_gen.add_argument("tables", help="path to xdsopl/LDPC dvb_s2_tables.hh")
    p_gen.add_argument("ids", nargs="*", help="table ids, e.g. B5 B7 (default: B5 B7)")
    p_gen.add_argument("--out", help="output directory (default: the shipped config/coding/block/ldpc)")
    p_gen.set_defaults(fn=cmd_gen_ldpc)

    args = parser.parse_args(argv)
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())

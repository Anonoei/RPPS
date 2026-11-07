"""
Example showing how to list available options
"""
import shutil
import textwrap

import rpps as rp

WIDTH = min(shutil.get_terminal_size((100, 20)).columns, 120)

def show(title, groups):
    """Print a title, then one wrapped, aligned line per group: {"group": [names]}"""
    print(f"\n{title}")
    if not groups:
        print("  (none)")
        return
    pad = max(len(k) for k in groups)
    for k, names in groups.items():
        lead = f"  {k:<{pad}}  "
        body = textwrap.fill(", ".join(names), WIDTH - len(lead), initial_indent=lead, subsequent_indent=" " * len(lead))
        print(body)

show("Modulation", rp.mod.ls())
show("Coding  (rp.coding.load(<kind>, \"<family>/<name>\"))", rp.coding.ls())
show("Scrambling", rp.scram.ls())
show("Encapsulation", rp.encap.ls())
show("Chains  (rp.Chain.load(\"<standard>/<name>\"))", rp._config.ls("chain"))
show("Filters  (rp.filters.load(<kind>, \"<family>/<name>\"))", rp.filters.ls())
show("Built-in Windows  (rp.filters.Window.get(<name>))", {"windows": rp.filters.Window.s()})

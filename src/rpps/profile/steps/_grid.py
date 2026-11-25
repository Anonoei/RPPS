"""grid steps (receive only): a resource grid's cell search, its frames, the channels read off them"""
from ...base.types import Rep
from ...errors import ConfigError
from ._core import STAGE, register_step, _given

@register_step("grid", "lte_cell")
def _lte_cell(kind, spec, rep, ctx):
    """LTE cell search and half-frame timing (rpps.grid.lte.LteCell): samples from the first subframe 0, in half-frames"""
    from ...grid import lte
    if rep is not None and rep != Rep.SAMPLES:
        raise ConfigError(f"grid lte_cell: takes samples, not {rep}")
    return [(STAGE, lte.LteCell(**_given(spec, ("offset", "span", "resolution", "acquire", "lose", "gate", "name"))))], [], Rep.SAMPLES

@register_step("grid", "lte_grid")
def _lte_grid(kind, spec, rep, ctx):
    """Subframes to grid frames with every port's channel (rpps.grid.lte.LteGrid)"""
    from ...grid import lte
    return [(STAGE, lte.LteGrid(**_given(spec, ("n_rb", "cell"))))], [], Rep.SYMBOLS

def _lte_channel(kind, spec, rep, ctx):
    """A channel on LTE grid frames (rpps.grid.lte.channels): gives bits, publishes what it decoded"""
    from ...grid import lte
    name = spec.get("id", kind)
    if kind == "lte_pbch":
        op = lte.PBCH(name=name)
    elif kind == "lte_pcfich":
        op = lte.PCFICH(name=name)
    elif kind == "lte_pdcch":
        op = lte.PDCCH(**_given(spec, ("rntis",)), name=name)
    else:
        op = lte.PDSCH(name=name)
    return [(STAGE, op)], [], Rep.DATA

for _k in ("lte_pbch", "lte_pcfich", "lte_pdcch", "lte_pdsch"):
    register_step("grid", _k)(_lte_channel)

@register_step("grid", "nr_cell")
def _nr_cell(kind, spec, rep, ctx):
    """NR cell search and slot timing, LEO Doppler tracked (rpps.grid.nr.NrCell): samples from a frame start, in slots"""
    from ...grid import nr
    if rep is not None and rep != Rep.SAMPLES:
        raise ConfigError(f"grid nr_cell: takes samples, not {rep}")
    return [(STAGE, nr.NrCell(**_given(spec, ("offset", "span", "resolution", "period", "acquire", "lose", "gate", "fc", "name"))))], [], Rep.SAMPLES

@register_step("grid", "nr_grid")
def _nr_grid(kind, spec, rep, ctx):
    """Slots to grid frames (rpps.grid.nr.NrGrid)"""
    from ...grid import nr
    return [(STAGE, nr.NrGrid(**_given(spec, ("cell", "period"))))], [], Rep.SYMBOLS

def _nr_channel(kind, spec, rep, ctx):
    """A channel on NR grid frames (rpps.grid.nr.channels)"""
    from ...grid.nr import channels as ch
    name = spec.get("id", kind)
    op = {"nr_pbch": lambda: ch.PBCH(name=name), "nr_pdcch": lambda: ch.PDCCH(**_given(spec, ("rntis",)), name=name),
          "nr_pdsch": lambda: ch.PDSCH(**_given(spec, ("fc",)), name=name)}[kind]()
    return [(STAGE, op)], [], Rep.DATA

for _k in ("nr_pbch", "nr_pdcch", "nr_pdsch"):
    register_step("grid", _k)(_nr_channel)

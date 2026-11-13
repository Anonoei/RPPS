"""Profiles: a whole signal as one config file, in receive order, both directions

    p = rp.profile.load("ieee802154/2450", sps=4)
    wave = p.tx()(rp.Data(payload))             # transmit: the transforms inverted, in reverse order
    bits = p.rx()(wave)                          # receive: every step as written
    bits = p.rx(start=rp.Rep.SYMBOLS)(symbols)   # a source that gives timed symbols: the steps before are skipped

config/profile/<family>/<mode>.json; see rpps.profile.steps for the step kinds, rpps.profile.control for header-driven
control (`controls`, `publish`, tables) and rpps.profile.runner for how runners apply it
"""
from .profile import Profile, load, ls
from .runner import Runner
from .parallel import Region
from .steps import Step
from .control import publish, anchor, table, Table, Control

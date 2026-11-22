"""Profiles: a whole signal as one config file, in receive order, both directions

    p = rp.profile.load("ieee802154/2450", sps=4)
    wave = p.tx()(rp.Data(payload))             # transmit: the transforms inverted, in reverse order
    bits = p.rx()(wave)                          # receive: every step as written
    bits = p.rx(start=rp.Rep.SYMBOLS)(symbols)   # a source that gives timed symbols: the steps before are skipped

What a profile needs (its rate, bandwidth, samples for a message) is `rp.profile.info(name)`, every profile's
`rp.profile.catalogue()`; what it decodes comes one frame a message (rp.profile.Message) from `runner.take_messages()`.

config/profile/<family>/<mode>.json; see rpps.profile.steps for the step kinds, rpps.profile.control for header-driven
control (`controls`, `publish`, tables) and rpps.profile.runner for how runners apply it
"""
from .profile import Profile, load, ls
from .message import Message
from .catalogue import ProfileInfo, info, catalogue
from .decoder import Decoder
from .runner import Runner
from .parallel import Region
from .steps import Step
from .control import table, Table

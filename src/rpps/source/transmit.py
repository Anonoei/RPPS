"""Transmitting with a radio: blocks of samples out of a pysdrlib device (`DeviceSink`)

    with rp.source.DeviceSink("hackrf", cf=433.92e6, Fs=2e6, tx_gain=0.3) as radio:
        enc.packet(pkt) >> radio                          # rp.profile.Encoder: a packet's samples

The device starts transmitting when the sink is made and stops when it is closed; `write` queues samples (complex, full
scale +-1) and blocks while the device's queue is full. A block's rate must be the sink's (be explicit: resample first,
or make the sink at the block's rate). Receive with rp.source.Device.
"""
import numpy as np

from .. import base
from . import formats as fmts

class DeviceSink:
    """Transmit blocks with a pysdrlib device: `block >> sink`, `blocks >> sink`

    device: a pysdrlib driver name ("hackrf"), a target string ("hackrf@SERIAL", "radiohost/hackrf-0": opened here, closed
    by `close()`), or a device you made (left open). cf, Fs: the transmit frequency and rate (None: the device's own).
    tx_gain: the transmit gain as a fraction 0-1 of the device's range. channel: the transmit channel. scale: multiplies
    the samples (headroom: clipping at +-1 is the radio's). count: samples queued"""
    def __init__(self, device, cf=None, Fs=None, tx_gain=None, channel=0, scale=1.0, **open_args):
        self._opened = False
        if isinstance(device, str):
            pysdrlib = fmts.pysdrlib()
            if "@" in device or "/" in device:
                device, open_args, self._opened = pysdrlib.open(device, **open_args), {}, True
            else:
                device = pysdrlib.device(device)
        self.sdr, self.channel, self.scale = device, channel, float(scale)
        if not self.sdr.state["open"]:
            self.sdr.open(**open_args)
            self._opened = True
        caps = getattr(self.sdr, "caps", None) # (what the device can do as opened: a sim made with transmit=True can)
        if not (caps.channels.get("tx") if caps is not None else getattr(self.sdr, "CAN_TRANSMIT", True)):
            name = getattr(type(self.sdr), "NAME", type(self.sdr).__name__)
            self.close()
            raise ValueError(f"DeviceSink: {name} can't transmit")
        if cf is not None:
            self.sdr.set_tx_freq(cf, channel=channel)
        if Fs is not None:
            self.sdr.set_tx_sample_rate(Fs, channel=channel)
        if tx_gain is not None:
            self.sdr.set_tx_gain(tx_gain, channel=channel)
        self.cf, self.Fs = self.sdr.get_tx_freq(channel=channel), self.sdr.get_tx_sample_rate(channel=channel)
        self.sdr.start_tx([channel])
        self.count = 0

    def write(self, data):
        """Queue a block (rp.Data at the sink's rate, or an array) for transmission; returns the sink"""
        if isinstance(data, base.data._Data):
            Fs = data.meta.Fs
            if Fs and self.Fs and abs(Fs - self.Fs) > 1e-6*self.Fs:
                raise ValueError(f"DeviceSink: a block at {Fs:g} Hz, the device transmits at {self.Fs:g} Hz (resample first)")
            data = data.data
        x = np.asarray(data, dtype=np.complex64)
        if self.scale != 1.0:
            x = x*np.float32(self.scale)
        if len(x):
            self.sdr.write(x, channel=self.channel)
            self.count += len(x)
        return self

    def __rrshift__(self, blocks):
        """blocks >> sink (a list or generator of blocks)"""
        for b in blocks:
            self.write(b)
        return self

    def close(self):
        if self.sdr is None:
            return
        try:
            self.sdr.stop_tx()
        finally:
            if self._opened and self.sdr.state["open"]:
                self.sdr.close()
            self.sdr = None

    def __enter__(self):
        return self
    def __exit__(self, *a):
        self.close()

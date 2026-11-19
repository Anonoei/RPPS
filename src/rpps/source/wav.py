"""WAV files in and out (the stdlib `wave` module: no dependency)

    rp.source.write_wav("out.wav", audio, 48000, peak=0.8)      # an array at once, scaled so its peak is 0.8 of full scale
    blocks + demod >> rp.source.WavSink("out.wav")             # a stream, at the blocks' own rate
    for blk in rp.source.Wav("sdr.wav", iq=True)(4096): ...    # in: 8/16/32-bit PCM; a 2-channel IQ recording as samples
"""
import wave

import numpy as np

from .source import Source

def _pcm(x, peak=None):
    """Samples as int16: floats in [-1, 1) (`peak`: first scaled so the largest is that fraction of full scale), clipped;
    integers as they are. Rows of a 2-D array are frames (columns the channels)"""
    x = np.asarray(x)
    if x.dtype.kind in "iu":
        return np.clip(x, -32768, 32767).astype(np.int16)
    x = np.real(x).astype(np.float64)
    if peak is not None:
        m = np.abs(x).max() if x.size else 0.0
        x = x*(peak/m) if m > 0 else x
    return np.clip(np.round(x*32767), -32768, 32767).astype(np.int16)

def write_wav(path, x, rate, peak=None):
    """Write audio (frames, or (frames, channels)) as a 16-bit WAV at `rate` Hz. peak: normalise to that fraction of full scale"""
    y = _pcm(x, peak)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1 if y.ndim == 1 else y.shape[1])
        w.setsampwidth(2)
        w.setframerate(int(round(rate)))
        w.writeframes(y.tobytes())
    return len(y)

class WavSink:
    """A 16-bit WAV file written block by block: `block >> sink` or `blocks >> sink`. Samples in [-1, 1) (clipped).
    rate: Hz, else the first block's Fs. channels: 1, or 2 for blocks of (frames, 2). Close it (or use `with`) to finish"""
    def __init__(self, path, rate=None, channels=1):
        self.path, self.rate, self.channels = str(path), rate, int(channels)
        self.count = 0
        self._w = None

    def write(self, data):
        x = np.asarray(getattr(data, "data", data))
        if self._w is None:
            rate = self.rate or getattr(data, "Fs", None)
            if not rate:
                raise ValueError("WavSink: give `rate`, or write Data that carries its Fs")
            self._w = wave.open(self.path, "wb")
            self._w.setnchannels(self.channels)
            self._w.setsampwidth(2)
            self._w.setframerate(int(round(rate)))
        y = _pcm(x)
        self._w.writeframes(y.tobytes())
        self.count += len(y)
        return self

    def close(self):
        if self._w is not None:
            self._w.close()
            self._w = None

    def __enter__(self):
        return self
    def __exit__(self, *a):
        self.close()


class Wav(Source):
    """A PCM WAV file as a source, scaled to [-1, 1): one channel gives real samples (audio); two give I and Q
    (`iq=True`: an SDR's IQ recording) or, with iq=False, the first channel. Fs is the file's rate"""
    def __init__(self, path, iq=None, cf=0.0):
        self.path = str(path)
        with wave.open(self.path, "rb") as w:
            self.channels, self.width, rate, self.frames = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        if self.width not in (1, 2, 4):
            raise ValueError(f"{path}: {8*self.width}-bit samples (8, 16 or 32-bit PCM)")
        self.iq = self.channels == 2 if iq is None else bool(iq)
        if self.iq and self.channels != 2:
            raise ValueError(f"{path}: IQ needs 2 channels, it has {self.channels}")
        super().__init__(float(rate), cf)

    def __str__(self):
        return f"Wav({self.path}, {self.channels} ch, {8*self.width}-bit, {self.Fs:g} Hz)"

    def _convert(self, raw):
        if self.width == 1: # 8-bit WAV is unsigned
            x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0)/128.0
        else:
            x = np.frombuffer(raw, f"<i{self.width}").astype(np.float32)/float(2**(8*self.width - 1))
        x = x.reshape(-1, self.channels)
        return (x[:, 0] + 1j*x[:, 1]).astype(np.complex64) if self.iq else x[:, 0].copy()

    def _samples(self, count):
        count = int(count)
        with wave.open(self.path, "rb") as w:
            left = self.frames
            while left > 0:
                n = min(count, left)
                left -= n
                yield self._block(self._convert(w.readframes(n)), flush=left == 0)

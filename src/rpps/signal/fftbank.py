"""FFTBank: many channels of one capture, each at its own width and rate, from one FFT per hop

    bank = rp.signal.FFTBank(Fs, [(mid, width), ...])      # Hz from the capture's centre, Hz wide
    for blk in src(16384):
        outs = bank.step(blk.data)                        # [samples at bank.Fs_out[k]] per channel, baseband
    outs = bank.flush()                                   # the rest, the end of the stream

Overlap-save on a fixed grid from the start of the stream (so the output does not depend on how the input is cut):
every `hop` input samples one FFT of `n` samples (the hop and `overlap` on each side). Each channel takes only the
bins around its centre, weighted by a smooth mask (flat to `guard` x width / 2, a raised-cosine skirt `transition` x
width wide), and an inverse FFT of n / D of them gives its samples decimated by D directly: D the largest power of two
that keeps `oversample` x width. The bin grid's rounding of the centre is put right with a phase ramp continuous across
hops, so a channel comes out at exactly `mid`. A signal_profile over twenty channels filters the capture once, where a
shift and decimation per channel (rp.signal.channelize) went over every sample once a channel.

The mask is zero phase (no delay); `n` is chosen so the skirt spans at least `min_bins` bins (its impulse response
stays inside the overlap), for the narrowest channel. `active` in `step` skips channels nobody reads any more.
"""
import numpy as np

def _pow2_at_least(x):
    return 1 << max(0, int(np.ceil(np.log2(max(x, 1)))))

class FFTBank:
    """Channels (mid, width) [Hz] of a capture at Fs: `step(x)` -> [baseband samples per channel]. Fs_out[k], D[k]"""

    def __init__(self, Fs, channels, oversample=4.0, guard=1.3, transition=0.2, min_bins=16, max_n=2**21):
        self.Fs = float(Fs)
        self.mid = np.array([float(c[0]) for c in channels])
        self.width = np.array([float(c[1]) for c in channels])
        if len(self.mid) == 0:
            raise ValueError("FFTBank: no channels")
        # Decimation: the largest power of two that keeps oversample x width (and at least 4 output samples a hop)
        self.D = [max(1, 1 << int(np.floor(np.log2(max(self.Fs/(oversample*w), 1.0))))) for w in self.width]
        need = max(min_bins*self.Fs/(transition*w) for w in self.width)
        self.n = int(min(max_n, max(_pow2_at_least(need), 8*max(self.D), 4096)))
        self.D = [min(D, self.n//8) for D in self.D]
        self.overlap = self.n//8
        self.hop = self.n - 2*self.overlap
        self.Fs_out = [self.Fs/D for D in self.D]
        self.bins, self.mask, self.k0, self.res = [], [], [], []
        for mid, width, D in zip(self.mid, self.width, self.D):
            m = self.n//D
            k0 = int(np.round(mid*self.n/self.Fs))
            off = np.fft.fftfreq(m)*m # 0..m/2-1, -m/2..-1: the bins around the centre, in IFFT order
            self.bins.append(((k0 + off).astype(np.int64)) % self.n)
            f = np.fft.fftfreq(m, D/self.Fs) + (k0*self.Fs/self.n - mid) # Each bin's offset from the channel's centre
            p = 0.5*guard*width
            s = min(p + transition*width, 0.5*self.Fs/D)
            a = np.abs(f)
            w = np.where(a <= p, 1.0, np.where(a >= s, 0.0, 0.5*(1 + np.cos(np.pi*(a - p)/max(s - p, 1e-12)))))
            self.mask.append((w*m/self.n).astype(np.complex64)) # (and the scale: n/D bins give n/D samples)
            self.k0.append(k0)
            self.res.append(mid/self.Fs - k0/self.n) # Cycles a sample the bin grid leaves to turn back
        self.buf = np.zeros(self.overlap, np.complex64) # The first hop's left overlap: before the stream, zeros
        self.start = -self.overlap # Absolute index of buf[0]
        self.seen = 0 # Input samples so far
        self.out = [0]*len(self.D) # Output samples given so far, per channel

    def __len__(self):
        return len(self.D)

    def _hop(self, x, which):
        X = np.fft.fft(x)
        s0 = self.start + self.overlap # Absolute index of the first sample this hop keeps
        outs = {}
        for k in which:
            D = self.D[k]
            y = np.fft.ifft(X[self.bins[k]]*self.mask[k])[self.overlap//D:(self.overlap + self.hop)//D]
            # The bins were taken around k0: y is x e^(-j 2 pi k0 (i - start)/n). Make it x e^(-j 2 pi mid i / Fs)
            i = s0 + D*np.arange(len(y), dtype=np.int64)
            turn = ((self.k0[k]*(self.start % self.n)) % self.n)/self.n + ((i*self.res[k]) % 1.0)
            outs[k] = (y*np.exp(-2j*np.pi*turn)).astype(np.complex64)
        return outs

    def step(self, x, active=None):
        """Samples in; [baseband samples per channel] out (empty arrays until a hop is in). active: the channels to
        compute (others give empty arrays and lose these samples)"""
        which = range(len(self.D)) if active is None else list(active)
        x = np.asarray(x, dtype=np.complex64)
        self.buf = np.concatenate((self.buf, x))
        self.seen += len(x)
        parts = {k: [] for k in which}
        while len(self.buf) >= self.n:
            for k, y in self._hop(self.buf[:self.n], which).items():
                parts[k].append(y)
            self.buf = self.buf[self.hop:]
            self.start += self.hop
        return self._gather(parts)

    def flush(self, active=None):
        """The end of the stream: the samples still held, zero padded, trimmed to the input's length"""
        which = range(len(self.D)) if active is None else list(active)
        end = self.seen
        parts = {k: [] for k in which}
        while self.start + self.overlap < end:
            seg = self.buf[:self.n]
            if len(seg) < self.n:
                seg = np.concatenate((seg, np.zeros(self.n - len(seg), np.complex64)))
            for k, y in self._hop(seg, which).items():
                parts[k].append(y)
            self.buf = self.buf[self.hop:]
            self.start += self.hop
        out = self._gather(parts)
        for k in which: # Only the outputs of samples that came in
            keep = max(0, -(-end//self.D[k]) - (self.out[k] - len(out[k])))
            self.out[k] -= len(out[k]) - min(keep, len(out[k]))
            out[k] = out[k][:keep]
        return out

    def _gather(self, parts):
        out = [np.zeros(0, np.complex64) for _ in self.D]
        for k, p in parts.items():
            if p:
                out[k] = np.concatenate(p)
                self.out[k] += len(out[k])
        return out

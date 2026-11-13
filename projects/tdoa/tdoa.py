"""TDOA (time difference of arrival) location of a transmitter from simultaneous captures at several known positions

    python projects/tdoa/tdoa.py synth                      # self-test on simulated receivers, clock offsets, drift and noise
    python projects/tdoa/tdoa.py measure a.cf64 b.cf64 --fs 8 --cf 407 --freq 407.6125 --seconds 30
    python projects/tdoa/tdoa.py locate sites.json          # solve a position from measured delays (see below)

Pieces (each importable on its own):
  coarse_fd     frequency difference of two copies (sampling clocks and local oscillators of separate radios differ by ppm), from the spectrum of a * conj(b)
  caf           cross-ambiguity function over delay and residual frequency, with a long coherent integration: the capture is cut into segments, each cross-correlated by FFT,
                the segments then summed coherently over frequency (and over a predicted delay drift: a clock error of 1 ppm moves the delay by 1 us per second). Delay peak refined
                well below a sample by evaluating the correlation on a fine delay grid
  measure       delay of a relative to b [s], frequency offset, peak-to-noise ratio, and the Cramer-Rao standard deviation for that SNR, bandwidth and time
  calibrate     remove each receiver's unknown start offset by measuring a reference transmitter of known position in the same capture
  locate        weighted least squares on the delay differences, with the 1-sigma error ellipse (from the Jacobian) and the receiver geometry (GDOP)

What the hardware must provide: radios that share a clock (GPS-disciplined reference on CLKIN, or one HackRF's CLKOUT into the others) so the delay is stable, and a known-position
transmitter on the air at the same time to calibrate each radio's start offset. Without a shared clock only the ppm drift is handled, and the calibration then has to be repeated often.
Delay precision is set by the signal's rms bandwidth: a 12.5 kHz channel gives metres at best in theory and hundreds of metres in practice (multipath); a wider signal is far better.
Meant to be upstreamed into RPPS.
"""
import argparse
import json
import sys

import numpy as np

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import _paths  # noqa: E402,F401
from rpps.math import least_squares
from rpps.signal.filters import low_pass

C = 299792458.0


# ---- estimation -----------------------------------------------------------------------------------------------------------------------------------------------------------------

def coarse_fd(a, b, fs):
    """Frequency of a relative to b [Hz]: peak of the spectrum of a * conj(b) (the common signal's modulation cancels), refined by a parabola"""
    z = a*np.conj(b); n = len(z); nf = 1 << int(np.ceil(np.log2(n))) + 1
    Z = np.abs(np.fft.fft(z*np.hanning(n), nf)); k = int(np.argmax(Z)); y0, y1, y2 = Z[k - 1], Z[k], Z[(k + 1) % nf]
    d = 0.5*(y0 - y2)/(y0 - 2*y1 + y2) if (y0 - 2*y1 + y2) != 0 else 0.0
    f = (k + d)*fs/nf
    return f - fs if f > fs/2 else f


def _segments(a, b, fs, seg):
    n = int(seg*fs); k = min(len(a), len(b))//n
    A = np.fft.fft(a[:k*n].reshape(k, n), 2*n, axis=1); B = np.fft.fft(b[:k*n].reshape(k, n), 2*n, axis=1)
    return A*np.conj(B), n, k


def caf(a, b, fs, max_delay=1e-3, seg=0.05, fd=None, drift=0.0, fine=0.02, span=2.0):
    """Delay of a relative to b [s] (positive: a hears the signal later), found with a coherent integration over the whole capture.
    fd: frequency of a minus b [Hz] (default coarse_fd); drift: delay change per second [s/s] to compensate.
    Returns dict(tau, fd, snr [dB, peak over the median of the surface], q [residual frequency bin used, Hz], surface (frequency x delay magnitude))"""
    if fd is None:
        fd = coarse_fd(a, b, fs)
    t = np.arange(len(a))/fs
    a = a*np.exp(-2j*np.pi*fd*t)  # removes the carrier difference, so that all segments add in phase
    X, n, k = _segments(a, b, fs, seg)
    f = np.fft.fftfreq(2*n, 1/fs)
    if drift:
        X = X*np.exp(2j*np.pi*f[None, :]*(drift*seg*np.arange(k))[:, None])  # shift each segment's correlation back by the delay drift accumulated so far
    c = np.fft.ifft(X, axis=1)
    m = int(max_delay*fs); lags = np.r_[np.arange(0, m + 1), np.arange(-m, 0)]
    c = c[:, lags]
    S = np.abs(np.fft.fft(c, axis=0))  # (residual-frequency bin) x lag
    q, i = np.unravel_index(np.argmax(S), S.shape)
    snr = 20*np.log10(S[q, i]/np.median(S))
    nu = np.fft.fftfreq(k, seg)[q]
    # fine delay: evaluate sum_f X e^{j 2 pi f tau}, summed over segments with the residual frequency phase, on a grid around the integer peak
    lag0 = lags[i]
    grid = lag0 + np.arange(-span, span + fine/2, fine)
    ph = np.exp(-2j*np.pi*nu*seg*np.arange(k))
    Xs = (ph[:, None]*X).sum(0)  # the coherent sum over segments (per frequency bin)
    val = np.array([np.abs((Xs*np.exp(2j*np.pi*f*g/fs)).sum()) for g in grid])
    j = int(np.argmax(val)); j = min(max(j, 1), len(grid) - 2)
    y0, y1, y2 = val[j - 1], val[j], val[j + 1]
    d = 0.5*(y0 - y2)/(y0 - 2*y1 + y2) if (y0 - 2*y1 + y2) != 0 else 0.0
    return dict(tau=float((grid[j] + d*fine)/fs), fd=float(fd + nu), snr=float(snr), q=float(nu), surface=S)


def crlb_sigma(snr_db, rms_bw, bw, t):
    """Cramer-Rao standard deviation of a delay estimate [s] for the per-sample SNR in bw, the signal's rms bandwidth, and the integration time"""
    snr = 10**(snr_db/10)
    return 1.0/(2*np.pi*rms_bw*np.sqrt(2*bw*t*snr))


def measure(a, b, fs, fc, **kw):
    """caf with the delay drift tried at -r, 0 and +r (r = fd/fc: radios locked to the same reference drift at the carrier offset ratio); the best peak wins"""
    fd = coarse_fd(a, b, fs); r = fd/fc if fc else 0.0
    best = None
    for dr in (0.0, -r, r):
        res = caf(a, b, fs, fd=fd, drift=dr, **kw); res["drift"] = dr
        if best is None or res["snr"] > best["snr"]:
            best = res
    return best


def calibrate(measured, ref_pos, ref_true_tdoa):
    """Receiver offsets [s] relative to the reference receiver: measured delay of the known transmitter minus its true delay. Subtract from later measurements"""
    return {k: measured[k] - ref_true_tdoa[k] for k in measured}


# ---- geometry ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def lla_to_enu(lat, lon, lat0, lon0):
    """Metres east and north of (lat0, lon0) on a local flat approximation (fine to tens of km)"""
    r = 6371000.0
    return np.array([np.radians(lon - lon0)*r*np.cos(np.radians(lat0)), np.radians(lat - lat0)*r])


def true_tdoa(pos, rx, ref=0):
    """Delay of every receiver relative to receiver `ref` for a transmitter at pos (ENU metres): (|p - r_i| - |p - r_ref|)/c"""
    d = np.linalg.norm(np.asarray(rx) - np.asarray(pos), axis=1)
    return (d - d[ref])/C


def locate(rx, tdoa, sigma, ref=0, x0=None):
    """(position ENU m, 2x2 covariance m^2, 1-sigma semi-axes m, residuals s). rx: Nx2 receiver positions; tdoa: N delays relative to receiver `ref` (the ref entry is ignored);
    sigma: N standard deviations [s]. N-1 delays fix a 2D position when N >= 3"""
    rx = np.asarray(rx, float); idx = [i for i in range(len(rx)) if i != ref]
    td = np.asarray(tdoa, float)[idx]; sg = np.asarray(sigma, float)[idx]
    x0 = rx.mean(0) if x0 is None else np.asarray(x0, float)

    def res(p):
        d = np.linalg.norm(rx - p, axis=1)
        return ((d[idx] - d[ref])/C - td)/sg
    best = None
    # the cost surface has hyperbola ambiguities: try the centroid and a ring of starts around the receivers
    span = max(np.ptp(rx[:, 0]), np.ptp(rx[:, 1]), 1000.0)
    starts = [x0] + [x0 + span*s*np.array([np.cos(a), np.sin(a)]) for s in (0.5, 2.0) for a in np.linspace(0, 2*np.pi, 8, endpoint=False)]
    for s in starts:
        r = least_squares(res, s)
        if best is None or r.cost < best.cost:
            best = r
    J = best.jac  # residuals are range differences / (c sigma): the Jacobian is already per metre of position, weighted
    cov = np.linalg.pinv(J.T@J)
    ev = np.sqrt(np.maximum(np.linalg.eigvalsh(cov), 0))
    return best.x, cov, ev[::-1], best.fun*sg


def gdop(rx, pos, ref=0):
    """Geometric dilution (metres of position error per metre of range-difference error) at pos for receivers rx"""
    rx = np.asarray(rx, float); d = np.linalg.norm(rx - pos, axis=1); u = (pos - rx)/d[:, None]
    G = np.array([u[i] - u[ref] for i in range(len(rx)) if i != ref])
    return float(np.sqrt(np.trace(np.linalg.inv(G.T@G))))


# ---- self-test ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _c4fm_like(n, fs, baud=4800, rng=None):
    """A band-limited 4-level FM signal (stand-in for a P25 channel), complex baseband"""
    rng = rng or np.random.default_rng(0)
    sps = fs/baud; sym = rng.choice([-3, -1, 1, 3], int(n/sps) + 4).astype(float)
    x = np.repeat(sym, int(np.ceil(sps)))[:n]; x = np.convolve(x, low_pass(int(sps*3) | 1, 0.6*baud, fs=fs).filt, "same")
    return np.exp(1j*2*np.pi*1800*np.cumsum(x)/3/fs)


def _receive(s, fs, delay, fd, drift, snr_db, rng):
    """Copy of s delayed by `delay` s (fractional, by a phase ramp), shifted by fd Hz, with the delay growing at `drift` s/s, plus noise"""
    n = len(s); f = np.fft.fftfreq(n, 1/fs); t = np.arange(n)/fs
    tau = delay + drift*t
    # constant part by FFT phase ramp; the drift part as a slow time warp by linear interpolation of the (oversampled) signal
    y = np.fft.ifft(np.fft.fft(s)*np.exp(-2j*np.pi*f*delay))
    if drift:
        y = np.interp(t - drift*t, t, y.real) + 1j*np.interp(t - drift*t, t, y.imag)
    y = y*np.exp(2j*np.pi*fd*t)
    p = np.mean(np.abs(y)**2); nz = (rng.standard_normal(n) + 1j*rng.standard_normal(n))*np.sqrt(p/10**(snr_db/10)/2)
    return y + nz


def synth(seed=1, snr_db=0.0, seconds=20.0, fs=50e3, fc=407.6125e6):
    rng = np.random.default_rng(seed)
    rx = np.array([[0.0, 0.0], [6000.0, 500.0], [2500.0, 5500.0], [-3000.0, 4000.0]])
    tx = np.array([1800.0, 1500.0]); ref_tx = np.array([-500.0, 3000.0])
    n = int(fs*seconds)
    sigs = {"target": _c4fm_like(n, fs, rng=rng), "reference": _c4fm_like(n, fs, rng=rng)}
    pos = {"target": tx, "reference": ref_tx}
    offs = np.array([0.0, 17.3e-6, -9.1e-6, 4.4e-6])  # unknown start offsets of each radio [s]
    ppm = np.array([0.0, 14.5, 16.0, 15.2])*1e-6
    caps = {}
    for name in sigs:
        caps[name] = []
        for i in range(len(rx)):
            dly = np.linalg.norm(rx[i] - pos[name])/C + offs[i]
            caps[name].append(_receive(sigs[name], fs, dly, fd=ppm[i]*fc, drift=ppm[i], snr_db=snr_db, rng=rng))
    ref_true = true_tdoa(ref_tx, rx)
    rows = {}
    for name in sigs:
        rows[name] = [0.0] + [measure(caps[name][i], caps[name][0], fs, fc, max_delay=200e-6)["tau"] for i in range(1, len(rx))]
    off = np.array([0.0] + [rows["reference"][i] - ref_true[i] for i in range(1, len(rx))])
    td = np.array(rows["target"]) - off
    sig = np.full(len(rx), crlb_sigma(snr_db, 2000.0, 8000.0, seconds)*10)  # 10x the bound: multipath and calibration are not free
    p, cov, ax, r = locate(rx, td, sig)
    print(f"{len(rx)} receivers, {seconds:.0f} s at {snr_db:+.0f} dB SNR, fs {fs/1e3:.0f} kS/s")
    print("true delays vs target [ns]:", np.round(true_tdoa(tx, rx)*1e9, 1))
    print("measured, calibrated  [ns]:", np.round(td*1e9, 1), " (offsets removed:", np.round(off*1e9, 0), ")")
    err = np.linalg.norm(p - tx)
    print(f"estimate {p.round(0)} vs true {tx}: error {err:.0f} m; 1-sigma axes {ax.round(0)} m; GDOP {gdop(rx, tx):.1f}")
    return err


# ---- command line ------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _channel(path, fs, cf, freq, seconds):
    import burst_tones
    x = np.fromfile(path, np.complex64, count=int(fs*seconds))
    return burst_tones.channel(x, fs, cf, freq)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth"); s.add_argument("--snr", type=float, default=0.0); s.add_argument("--seconds", type=float, default=20.0); s.add_argument("--seed", type=int, default=1)
    m = sub.add_parser("measure", help="delay of file A relative to file B for one channel")
    m.add_argument("a"); m.add_argument("b"); m.add_argument("--fs", type=float, required=True, help="[MS/s]"); m.add_argument("--cf", type=float, required=True, help="[MHz]")
    m.add_argument("--freq", type=float, required=True, help="channel [MHz]"); m.add_argument("--seconds", type=float, default=30.0); m.add_argument("--max-delay", type=float, default=200.0, help="[us]")
    lo = sub.add_parser("locate", help='JSON: {"ref": 0, "receivers": [{"lat":..,"lon":..}], "tdoa_ns": [0, ..], "sigma_ns": [.., ..]}')
    lo.add_argument("file")
    a = ap.parse_args(argv)
    if a.cmd == "synth":
        synth(seed=a.seed, snr_db=a.snr, seconds=a.seconds)
    elif a.cmd == "measure":
        fs = a.fs*1e6; za, fs1 = _channel(a.a, fs, a.cf*1e6, a.freq*1e6, a.seconds); zb, _ = _channel(a.b, fs, a.cf*1e6, a.freq*1e6, a.seconds)
        r = measure(za, zb, fs1, a.freq*1e6, max_delay=a.max_delay*1e-6)
        print(f"delay of A relative to B: {r['tau']*1e9:.0f} ns ({r['tau']*C:.0f} m of range difference); frequency difference {r['fd']:.2f} Hz ({r['fd']/(a.freq*1e6)*1e6:.2f} ppm); "
              f"drift {r['drift']*1e6:.2f} us/s; peak {r['snr']:.1f} dB over the surface median")
    else:
        d = json.load(open(a.file)); rec = d["receivers"]; ref = d.get("ref", 0)
        rx = np.array([lla_to_enu(r["lat"], r["lon"], rec[ref]["lat"], rec[ref]["lon"]) for r in rec])
        p, cov, ax, res = locate(rx, np.array(d["tdoa_ns"])*1e-9, np.array(d["sigma_ns"])*1e-9, ref=ref)
        print(f"position ENU from the reference receiver: east {p[0]:.0f} m, north {p[1]:.0f} m; 1-sigma axes {ax.round(0)} m; residuals {np.round(res*1e9, 0)} ns")


if __name__ == "__main__":
    main()

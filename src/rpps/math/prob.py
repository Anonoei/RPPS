"""Probability and statistics helpers"""
import numpy as np

def logsumexp(x, axis=-1):
    """log(sum(exp(x))) along an axis, without overflow"""
    x = np.asarray(x)
    m = x.max(axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0) # All -inf: the sum is 0 and the log -inf, not nan
    with np.errstate(divide="ignore"):
        return (m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))).squeeze(axis)

def z_critical(trials, alpha=0.01):
    """z a null hypothesis exceeds with probability alpha/trials (Gaussian tail estimate)"""
    return float(np.sqrt(2*np.log(max(trials, 2)/alpha)))

def log_binom_half_cdf(n, e):
    """log P(Binomial(n, 1/2) <= e) for every e in an array (or one), exactly, in logs (no overflow at any n): the chance
    a random n-bit window is within e errors of a word. -inf below 0"""
    from math import lgamma, log
    e = np.atleast_1d(np.asarray(e, dtype=np.int64))
    top = int(min(max(e.max(initial=-1), -1), n))
    if top < 0:
        out = np.full(e.shape, -np.inf)
    else:
        k = np.arange(top + 1)
        lt = np.array([lgamma(n + 1) - lgamma(j + 1) - lgamma(n - j + 1) for j in k]) - n*log(2) # log P(exactly k)
        cum = np.logaddexp.accumulate(lt)
        out = np.where(e < 0, -np.inf, cum[np.clip(e, 0, top)])
    return out if out.size > 1 else float(out[0])

def deviance_z(k, n, p0=0.5):
    """Signed z-like score of k events in n trials against probability p0 (binomial deviance)"""
    k = np.asarray(k, dtype=np.float64)
    mu = n*p0
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(k > 0, k*np.log(k/mu), 0.0) + np.where(k < n, (n-k)*np.log((n-k)/(n-mu)), 0.0)
    return np.sign(k - mu) * np.sqrt(2*np.maximum(d, 0))

def mixture_variance(d2, iters=8, floor=1e-12):
    """Noise variance per real dimension of observations scattered around equiprobable points, by EM

    d2: (N, P) squared distances of N observations from P points. Each observation is a Gaussian mixture draw
    over the points with one shared variance, which is what the iterations estimate.
    """
    var = np.mean(np.min(d2, axis=1))/2 + floor
    for _ in range(iters):
        logp = -d2/(2*var)
        r = np.exp(logp - logp.max(axis=1, keepdims=True))
        r /= r.sum(axis=1, keepdims=True)
        var = max(np.sum(r*d2)/(2*len(d2)), floor)
    return var

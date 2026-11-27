"""Is a bit stream as random as a cipher's output? NIST SP 800-22 tests, and byte-level entropy

    r = rp.analysis.framing.randomness.test(bits)
    r.p                    # {test: P-value}: frequency, block_frequency, runs, serial_1, serial_2, approximate_entropy,
                           #   cusum_forward, cusum_backward, byte_chi2
    r.passed, r.failed     # each against alpha / (number of P-values): a random stream fails one test in 100 runs
    r.entropy              # Shannon entropy of the bytes [bits a byte], 8 at most
    r.verdict              # "random" (indistinguishable from a cipher's output), "high entropy" (fails some test,
                           #   its bytes near uniform), or "low entropy"

The tests are NIST SP 800-22 rev. 1a (section 2), as the standard defines them, each checked against its worked examples:
frequency (2.1), frequency within a block (2.2), runs (2.3), serial (2.11), approximate entropy (2.12) and cumulative sums
(2.13), plus a chi-square test of the 256 byte values. Entropy alone cannot tell encrypted from compressed data, both
are near 8 bits a byte; the tests that work best at it are block frequency, cumulative sums, serial and approximate
entropy, and the byte chi-square (De Gaspari et al., "Reliable detection of compressed and encrypted data", 2022; HEDGE,
2019; Davies et al., "Comparison of entropy calculation methods for ransomware encrypted file identification", 2022).
So a stream that passes every test is called "random": encrypted, keyed (a stream cipher), or a true random source; one
with high byte entropy that fails some test is "high entropy": compressed, or a weak scrambler or cipher. A short stream
cannot tell either well: the tests' input-size recommendations are kept (each test needs at least 100 bits, the block,
serial and entropy lengths grow with n).
"""
from math import erfc, log, sqrt

import numpy as np

from ...math.prob import igamc

ALPHA = 0.01

def _phi(x):
    """Standard normal CDF"""
    return 0.5*erfc(-x/sqrt(2.0))

def frequency(eps):
    """2.1: the proportion of ones"""
    n = len(eps)
    s = abs(2*int(eps.sum()) - n)/sqrt(n)
    return erfc(s/sqrt(2.0))

def block_frequency(eps, M=None):
    """2.2: the proportion of ones in M-bit blocks (M >= 20, M > n/100, fewer than 100 blocks)"""
    n = len(eps)
    M = M or max(20, n//99 + 1)
    N = n//M
    pi = eps[:N*M].reshape(N, M).mean(axis=1)
    chi = 4.0*M*float(((pi - 0.5)**2).sum())
    return igamc(N/2.0, chi/2.0)

def runs(eps):
    """2.3: the number of runs of identical bits (0 when the frequency prerequisite fails)"""
    n = len(eps)
    pi = float(eps.mean())
    if abs(pi - 0.5) >= 2.0/sqrt(n):
        return 0.0
    v = 1 + int(np.count_nonzero(eps[1:] != eps[:-1]))
    return erfc(abs(v - 2*n*pi*(1 - pi))/(2*sqrt(2*n)*pi*(1 - pi)))

def _pattern_counts(eps, m):
    """Counts of the n overlapping m-bit patterns of eps with its first m-1 bits appended (wrapped)"""
    if m == 0:
        return np.array([len(eps)])
    e = np.concatenate((eps, eps[:m - 1])).astype(np.int64)
    v = np.zeros(len(eps), dtype=np.int64)
    for k in range(m):
        v = (v << 1) | e[k:k + len(eps)]
    return np.bincount(v, minlength=1 << m)

def _psi2(eps, m):
    if m <= 0:
        return 0.0
    n = len(eps)
    c = _pattern_counts(eps, m).astype(np.float64)
    return (2.0**m/n)*float((c*c).sum()) - n

def serial(eps, m=None):
    """2.11: every overlapping m-bit pattern equally often (m < log2 n - 2). Returns (P-value 1, P-value 2)"""
    n = len(eps)
    m = m or max(2, min(16, int(np.log2(n)) - 3))
    p0, p1, p2 = _psi2(eps, m), _psi2(eps, m - 1), _psi2(eps, m - 2)
    return igamc(2.0**(m - 2), (p0 - p1)/2.0), igamc(2.0**(m - 3), (p0 - 2*p1 + p2)/2.0)

def approximate_entropy(eps, m=None):
    """2.12: the frequencies of overlapping m- and (m+1)-bit patterns (m < log2 n - 5)"""
    n = len(eps)
    m = m or max(2, min(14, int(np.log2(n)) - 6))
    def phi(k):
        c = _pattern_counts(eps, k)/n
        c = c[c > 0]
        return float((c*np.log(c)).sum())
    apen = phi(m) - phi(m + 1)
    return igamc(2.0**(m - 1), n*(log(2.0) - apen))

def cusum(eps, backward=False):
    """2.13: the largest excursion of the random walk of +-1 (mode 0 forward, mode 1 backward)"""
    n = len(eps)
    x = 2*eps.astype(np.int64) - 1
    z = int(np.abs(np.cumsum(x[::-1] if backward else x)).max())
    if z == 0:
        return 1.0
    sn = sqrt(n)
    s1 = sum(_phi((4*k + 1)*z/sn) - _phi((4*k - 1)*z/sn) for k in range(int((-n/z + 1)/4), int((n/z - 1)/4) + 1))
    s2 = sum(_phi((4*k + 3)*z/sn) - _phi((4*k + 1)*z/sn) for k in range(int((-n/z - 3)/4), int((n/z - 1)/4) + 1))
    return 1.0 - s1 + s2

def byte_chi2(eps):
    """Chi-square of the 256 byte values against uniform (255 degrees of freedom)"""
    B = np.packbits(eps[:len(eps)//8*8])
    c = np.bincount(B, minlength=256).astype(np.float64)
    e = len(B)/256.0
    return igamc(127.5, float(((c - e)**2).sum()/e)/2.0)

def byte_entropy(eps):
    """Shannon entropy of the bytes [bits a byte]"""
    B = np.packbits(eps[:len(eps)//8*8])
    c = np.bincount(B, minlength=256).astype(np.float64)
    q = c[c > 0]/len(B)
    return float(-(q*np.log2(q)).sum())

class Result:
    """`test`: p {test: P-value}, alpha (per test, after the multiple-test correction), passed, failed [test names],
    entropy (bits a byte), n (bits tested), verdict"""
    __slots__ = ("p", "alpha", "passed", "failed", "entropy", "n", "verdict")
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __str__(self):
        worst = min(self.p, key=self.p.get)
        f = f"; fails {', '.join(self.failed)}" if self.failed else ""
        return (f"{self.verdict}: {len(self.p) - len(self.failed)} of {len(self.p)} NIST SP 800-22 / byte tests pass "
                f"(lowest P {self.p[worst]:.2g}, {worst}){f}, {self.entropy:.3f} bits a byte, {self.n} bits")

    __repr__ = __str__

def test(bits, alpha=ALPHA, max_bits=1_000_000):
    """Every test on (the first `max_bits` of) the bits. A test fails below alpha / (number of P-values), so a random
    stream fails any one in about 1 run of 1/alpha. Fewer than 1000 bits: None (too short to say)"""
    eps = np.asarray(bits, dtype=np.uint8)[:max_bits] & 1
    n = len(eps)
    if n < 1000:
        return None
    s1, s2 = serial(eps)
    p = {"frequency": frequency(eps), "block_frequency": block_frequency(eps), "runs": runs(eps), "serial_1": s1,
         "serial_2": s2, "approximate_entropy": approximate_entropy(eps), "cusum_forward": cusum(eps),
         "cusum_backward": cusum(eps, backward=True), "byte_chi2": byte_chi2(eps)}
    a = alpha/len(p)
    failed = [k for k, v in p.items() if v < a]
    H = byte_entropy(eps)
    full = 8.0 - 255.0/(2.0*(n//8)*log(2.0)) # What uniform bytes give from this many (the plug-in estimate falls short)
    verdict = "random" if not failed else ("high entropy" if H > full - 0.1 else "low entropy")
    return Result(p=p, alpha=a, passed=not failed, failed=failed, entropy=H, n=n, verdict=verdict)

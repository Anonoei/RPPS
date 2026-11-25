"""Reference values and implementations the tests compare against"""
import numpy as np

from rpps.signal.filters import iir

def lfilter_ref(b, a, x):
    """y[n] = (sum b[k] x[n-k] - sum a[k] y[n-k]) / a[0], one sample at a time"""
    b, a = np.asarray(b, float), np.asarray(a, float)
    y = np.zeros(len(x), dtype=np.result_type(x, float))
    for n in range(len(x)):
        acc = sum(b[k]*x[n-k] for k in range(len(b)) if n-k >= 0) - sum(a[k]*y[n-k] for k in range(1, len(a)) if n-k >= 0)
        y[n] = acc / a[0]
    return y

def gain_db(sos, f):
    """Magnitude (dB) at the frequency f (cycles per sample)"""
    z = np.exp(-2j*np.pi*f)
    H = np.prod([(b0 + b1*z + b2*z**2)/(a0 + a1*z + a2*z**2) for b0, b1, b2, a0, a1, a2 in iir.as_sos(sos)])
    with np.errstate(divide="ignore"):
        return 20*np.log10(abs(H))

# MacKay alist of the Hamming (7,4) parity checks: 7 bits, 3 checks
HAMMING_ALIST = """7 3
3 4
1 1 2 2 1 2 3
4 4 4
1 0 0
2 0 0
1 2 0
1 3 0
3 0 0
2 3 0
1 2 3
1 3 4 7
2 3 6 7
4 5 6 7
"""

"""Phase Locked Loop helpers

Composed of:
- Error detector, (eD[n]), error detector output
- Loop filter,    (eF[n]), filter output
- Generator,      (eC[n]), estimate
  - Accumulator
    - eC[n] = (eC[n-1] + K0 * eF[n-1] mod [max])

`loop_gains` designs the filter from a bandwidth, which is how Costas and the timing loops should be tuned.
"""

def loop_gains(bw, damping=0.7071, Kd=1.0, K0=1.0):
    """Proportional and integral gains (alpha, beta) of a second order loop

    bw: loop noise bandwidth, normalised to the rate the loop runs at (0.01-0.05 is typical: 1-5 % of the rate)
    damping: damping factor (1/sqrt(2); 0.5-2.0)
    Kd: detector gain, K0: generator gain
    """
    theta = bw/(damping + 1/(4*damping))
    den = 1 + 2*damping*theta + theta**2
    return 4*damping*theta/den/(Kd*K0), 4*theta**2/den/(Kd*K0)

def K_fs(Kd, K0, Df, Bn, Fs):
    """(Kp, Ki) for a loop of noise bandwidth Bn [Hz] running at Fs [Hz]"""
    Kp = (1/(Kd*K0)) * ((4*Df)/(Df+(1/(4*Df)))) * (Bn/Fs)
    Ki = (1/(Kd*K0)) * (4/(Df+(1/(4*Df)))**2) * (Bn/Fs)**2
    return Kp, Ki

def K_sr(Kd, K0, Df, Bn, sr, sps):
    """(Kp, Ki) for a loop of noise bandwidth Bn [Hz] that updates once per symbol (sr: symbol rate [Hz])"""
    Kp = (1/(Kd*K0*sps)) * ((4*Df)/(Df+(1/(4*Df)))) * (Bn/sr)
    Ki = (1/(Kd*K0*sps**2)) * (4/(Df+(1/(4*Df)))**2) * (Bn/sr)**2
    return Kp, Ki

class PILoop:
    """Proportional-integral loop filter: eF[n] = Kp*eD[n] + (sum of Ki*eD)"""
    __slots__ = ("Kp", "Ki", "integ")
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.integ = 0.0

    @classmethod
    def from_bandwidth(cls, bw, damping=0.7071, Kd=1.0, K0=1.0):
        return cls(*loop_gains(bw, damping, Kd, K0))

    def run(self, error):
        self.integ += self.Ki * error
        return self.Kp * error + self.integ

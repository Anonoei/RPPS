"""Phase Locked Loop

Composed of:
- Error detector, (eD[n]), error detector output
- Loop filter,    (eF[n]), filter output
- Generator,      (eC[n]), estimate
  - Accumulator
    - eC[n] = (eC[n-1] + K0 * eF[n-1] mod [max])
"""

class DPLL:
    pass

def K_fs(Kd, K0, Df, Bn, Fs):
    Kp = (1/(Kd*K0)) * ((4*Df)/(Df+(1/(4*Df)))) * (Bn/Fs)
    Ki = (1/(Kd*K0)) * (4/(Df+(1/(4*Df)))**2) * (Bn/Fs)**2
    return Kp, Ki

def K_sr(Kd, K0, Df, Bn, sr, sps):
    Kp = (1/(Kd*K0*sps)) * ((4*Df)/(Df+(1/(4*Df)))) * (Bn/sr)
    Ki = (1/(Kd*K0*sps**2)) * (4/(Df+(1/(4*Df)))**2) * (Bn/sr)**2

class DPLL_PI:
    """
    P, eF1[n] = Kp * eD[n]
    U, eF2[n] = eF2[n-1] + Ki * eD[n]
    eF[n] = eF1[n] + eF2[n]

    Df, ζ [1/sqrt(2), 0.707, 0.5-2.0]
    Bn, ωn [1-5% Fs or sr]
    K0 = [1], Generator gain
    Kd = [A/2, 0.5], Error gain
    Kp
    """
    def __init__(self):
        self._int = 0.0

    def run(self, error):
        self._int += self.Ki * error
        self.err = self.Kp * error + self._int
        self.ref = self.ref + self.K0 * self.err

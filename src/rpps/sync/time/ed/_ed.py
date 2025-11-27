from ...ed import Type

import numpy as np

class _TED:
    Type = Type.NONE
    iSPS = 2 # input samples per symbol
    Inputs = 3 # sample count to calculate error
    Lookahead = False # Lookahead sample
    Derivative = False # Derivative input sample

    def __init__(self):
        self.err = 0.0
        self.prev_err = 0.0
        self.clock = 0
        self.samples = np.zeros(self.Inputs)

    def run(self):
        ...

    def input(self, sample):
        ...

    def _clock_adv(self):
        self.clock = (self.clock + 1) % self.Inputs
    def _clock_rev(self):
        if self.clock == 0:
            self.clock = self.Inputs - 1
        else:
            self.clock -= 1
    def _clock_reset(self):
        self.clock = self.Inputs - 1

    def revert(self, preserve: bool):
        if self.clock == 0 and not preserve:
            self.err = self.prev_err
        self._clock_rev()

    def sync_reset(self):
        self.err = 0.0
        self.prev_err = 0.0

class TED_ND(_TED):
    Type = Type.ND

    def input(self, sample):
        self.samples[self.clock] = sample
        self._clock_adv()

        if self.clock == 0 and not self.Lookahead:
            self.prev_err = self.err
            self.err = self.run()

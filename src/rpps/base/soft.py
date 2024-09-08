"""Soft decision helpers"""
import numpy as np

class SoftDecision:
    """SoftDecision type"""
    def __init__(self, codewords=None, probabilities=None):
        if codewords is None:
            codewords = np.zeros((1, 2))
        if probabilities is None:
            probabilities = np.zeros((1, 2))
        self.codewords = codewords
        self.probabilities = probabilities
        self._bits = None
        self._hard = None
        self.decided = False

    @property
    def bits(self):
        """Return decided bits, otherwise hard bits"""
        if self.decided:
            return self._bits
        return self.hard()

    def decide(self, code_idx):
        """Decide the codeword for the next bits index"""
        if self._bits is None:
            self._bits = np.zeros((1, self.codewords.shape[1]), dtype=bool)
            self._bits[0] = self.codewords[code_idx]
            return self._bits[-1]
        elif self._bits.shape[0] < self.probabilities.shape[0]:
            self._bits = np.append(self._bits, self.codewords[code_idx], axis=1)
            if self._bits.shape[0] == self.probabilities.shape[0]:
                self.decided = True
            return self._bits[-1]
        return None

    def hard(self):
        """Perform a hard decision from probabilities"""
        if self._hard is None:
            max_vals = np.max(self.probabilities, axis=1)
            indices = np.argwhere(np.equal(self.probabilities, max_vals[:, None]))
            self._hard = self.codewords[indices[:,1]].reshape(-1).astype(bool)
        return self._hard

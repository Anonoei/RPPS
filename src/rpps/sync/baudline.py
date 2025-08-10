import numpy as np
import matplotlib.pyplot as plt

from ..filters.impl import window
def find_rate(syms):
    """Find sps from IQ samples"""
    # m_syms = syms * np.conj(syms)
    # rline_syms = np.zeros(len(m_syms) * 3)
    # rline_syms[::3] = np.abs(m_syms)

    # baudline = np.fft.fft(rline_syms)
    # rline = np.argmax(baudline[10:len(baudline)//2]) + 10
    syms = syms**4
    syms *= window.blackman(len(syms))
    frq = np.fft.fftshift(np.fft.fft(syms))
    frq = 10*np.log10(frq)
    plt.plot(frq)
    plt.show()
    # exit()
    rline = np.argmax(frq[10:len(frq)//2]) + 10
    return rline/len(syms)

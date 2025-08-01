"""Rate processing"""
import numpy as np

import matplotlib.pyplot as plt

def find_rate(syms):
    """Find symbol rate from IQ samples"""
    m_syms = syms * np.conj(syms)
    rline_syms = np.zeros(len(m_syms) * 3)
    rline_syms[::3] = np.abs(m_syms)

    baudline = np.fft.fft(rline_syms)
    plt.plot(baudline)
    plt.show()
    line_val = max(baudline)
    idx = np.where(baudline == line_val)
    print(f"Got baudline {line_val} -- {idx}\n")
    return idx[0][1], max(baudline)

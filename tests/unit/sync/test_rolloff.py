"""What detect tries: the PSK, QAM and APSK definitions, and the roll-off the spectrum's edges give (rp.sync.rate.find_rolloff)"""
import numpy as np
import pytest

import rpps as rp
from rpps.sync import acquire as A
from rpps.sync.rate import find_rolloff, nearest_rolloff

from support.signals import shaped

def test_the_candidates_are_the_families_own_point_sets():
    names = set(A.candidates())
    assert {"bpsk", "qpsk", "8psk", "16qam", "32qam", "16apsk", "32apsk"} <= names
    assert not {n for n in names if "dvbs2" in n or n == "qpsk_axes" or n == "4level"} # children, and real levels, aren't

@pytest.mark.parametrize("beta", [0.05, 0.1, 0.2, 0.35])
def test_the_edges_give_the_standard_rolloff(beta):
    """8192 symbols at 10 dB, five draws each of QPSK and 16APSK: within 0.06 and nearest the right standard one"""
    got = []
    for seed in range(5):
        for mod in ("qpsk", "16apsk"):
            x, _ = shaped(np.random.default_rng(seed), A.candidates([mod])[mod], 8192, 4, beta=beta, esn0_db=10, span=48)
            got.append(find_rolloff(x, 4e6, 1e6)[0])
    assert abs(np.mean(got) - beta) < 0.06 and nearest_rolloff(np.median(got)) == beta, got

@pytest.mark.parametrize("mod", ["16qam", "32qam", "16apsk", "32apsk"])
def test_detect_names_qam_and_apsk(mod):
    x, _ = shaped(np.random.default_rng(1), A.candidates([mod])[mod], 4096, 4, beta=0.2, esn0_db=18, cfo=2e-4, span=24)
    assert A.detect(x, 4e6, baud=1e6).mod == mod

def test_a_small_rolloff_is_matched():
    """DVB-S2X's 0.05: the likelihood alone would take 0.15 (it can't tell them); the edges bring 0.05 in"""
    x, _ = shaped(np.random.default_rng(2), A.candidates(["8psk"])["8psk"], 8192, 4, beta=0.05, esn0_db=15, span=48)
    det = A.detect(x, 4e6, baud=1e6)
    assert det.mod == "8psk" and det.beta <= 0.1 and det.rolloff < 0.1
    assert A.detect(x, 4e6, baud=1e6, betas=(0.2, 0.25, 0.35)).rolloff is None # (given betas: nothing measured)

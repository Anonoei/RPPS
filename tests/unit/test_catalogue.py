"""Measured features to named systems (rpps.analysis.catalogue)"""
from rpps.analysis import catalogue as C

def test_gsm_from_a_cpm_result_and_tdma_frame():
    f = C.features(cpm="gmsk_0.3", baud=270830, frame=4.6153e-3, slots=8, frequency=947e6)
    m = C.match(f)
    assert m[0].name == "GSM" and not m[0].contradict

def test_lte_from_pilots():
    m = C.match(dict(modulation="ofdm", scs=14990, cp=4.69e-6, pilot_period=140, pilot_spacing=6))
    assert m[0].name == "LTE"

def test_more_agreement_ranks_higher():
    m = C.match(dict(modulation="8psk", baud=18000, frame=56.7e-3))
    assert m[0].name == "TETRA" and m[0].score > m[1].score

def test_contradictions_are_reported():
    m = {x.name: x for x in C.match(dict(modulation="gmsk", baud=9600, bt=0.4))}
    assert m["AIS"].score > 0.5 and any(f == "baud" for f, _ in m["GSM"].contradict)

def test_tool_names_fold_to_families():
    assert C.family("soqpsk_tg") == "soqpsk" and C.family("4-GFSK") == "4fsk"

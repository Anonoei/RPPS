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

# ---- Satellite and cellular cases: ties reported, weak evidence no match, features that break ties

def test_dvbs2_keeps_a_continuous_8psk():
    m = C.match(dict(modulation="8psk", baud=20e6, beta=0.2, frequency=11.5e9))
    assert m[0].name in ("DVB-S2", "DVB-S2X") and m[0].score > 0.5

def test_a_small_roll_off_is_dvbs2x():
    assert C.match(dict(modulation="qpsk", baud=30e6, beta=0.05, frequency=11.5e9))[0].name == "DVB-S2X"

def test_equal_fits_are_reported_as_ambiguous_and_a_profile_breaks_them():
    m = C.match(dict(modulation="qpsk", baud=27.5e6, beta=0.35, frequency=11.5e9))
    assert {"DVB-S", "DVB-S2"} <= {m[0].name, *m[0].tied} and "ambiguous" in str(m[0])
    m = C.match(dict(modulation="qpsk", baud=27.5e6, beta=0.35, frequency=11.5e9, profile="dvbs2/normal"))
    assert m[0].name == "DVB-S2" and not m[0].tied

def test_scpc_and_return_bursts():
    m = C.match(dict(modulation="qpsk", baud=2.048e6/1.5, burst=False, frequency=11.5e9))
    assert m[0].name.startswith("SCPC")
    m = C.match(dict(modulation="qpsk", baud=512e3, frame=0.1, burst=True, frequency=11.5e9))
    assert m[0].name.startswith("DVB-RCS")

def test_pm_ttc_in_c_and_ku_is_not_against_its_band():
    for f in (3.8e9, 11.5e9):
        m = {x.name: x for x in C.match(dict(modulation="pm", frequency=f))}
        assert any(n.startswith("PM TT&C") and not x.contradict for n, x in m.items())
    m = C.match(dict(modulation="pm", frequency=11.5e9, subcarrier=65.536e3, subcarrier_baud=4096))
    assert m[0].name.startswith("PM TT&C") and m[0].score > 0.5

def test_weak_evidence_is_no_match():
    assert C.match(dict(modulation="bpsk", baud=19.2e3, frequency=11.5e9))[0].score <= 0
    assert all(x.score <= 0 for x in C.match(dict(frequency=3.7e9)))
    m = C.match(dict(modulation="cw", frequency=3.7e9))
    assert m[0].name.startswith("Unmodulated") and m[0].score > 0

def test_sync_signals_and_ssb_periods_tell_lte_from_nr():
    base = dict(modulation="ofdm", scs=15e3, cp=144/2048/15e3, frame=5e-3, frequency=3.5e9)
    assert {"LTE", "5G NR"} <= {C.match(base)[0].name, *C.match(base)[0].tied}
    assert C.match({**base, "sync": "pss_sss"})[0].name == "LTE"
    assert C.match({**base, "sync": "ssb"})[0].name == "5G NR"
    m = C.match(dict(modulation="ofdm", scs=30e3, cp=144/2048/30e3, frame=20e-3, frequency=3.7e9))
    assert m[0].name == "5G NR" and not m[0].contradict

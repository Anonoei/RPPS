"""Keys nothing reads are reported (rpps.errors.ConfigWarning): a misspelling, or a setting of another method"""
import warnings

import pytest

import rpps as rp
from rpps.errors import ConfigWarning

def test_a_definition_key_nothing_reads_warns():
    obj = {"base": "fir", "type": "windowed", "pass": "lowpass", "cutof": 0.1, "cutoff": 0.1, "taps": 31}
    with pytest.warns(ConfigWarning, match=r"filter/t/typo: 'cutof' not used"):
        rp._config.load_obj("filter", "t/typo", obj)

def test_a_design_made_later_still_has_its_keys_counted():
    """A windowed FIR is designed when the stream starts (its rate): what it will read is taken at load"""
    obj = {"base": "fir", "type": "windowed", "pass": "lowpass", "cutoff": 0.1, "transition": 0.02, "atten_db": 60}
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConfigWarning)
        rp._config.load_obj("filter", "t/fine", obj)

def _profile(steps, defaults=None, sync=None):
    return rp.profile.Profile("t/keys", {"steps": steps, **({"defaults": defaults} if defaults else {})}, sync=sync)

SHAPED = [{"wave": "shape", "pulse": "rrc/0p35", "sps": 4}, {"sync": "timing"}]

def test_a_step_key_nothing_reads_warns():
    with pytest.warns(ConfigWarning, match=r"step 'carrier': 'alpha' not used"): # (a loop's gain: MPower has none)
        _profile(SHAPED + [{"sync": "carrier", "method": "mpower", "order": 4, "alpha": 0.01}])

def test_a_default_its_method_does_not_take_warns():
    with pytest.warns(ConfigWarning, match=r"defaults 'timing': 'alpha' not used by its method"):
        _profile(SHAPED, defaults={"timing": {"method": "oerder_meyr", "alpha": 0.3}})

def test_defaults_for_a_step_the_profile_does_not_have_warn():
    with pytest.warns(ConfigWarning, match=r"defaults 'equalize': no equalize step takes \['taps'\]"):
        _profile(SHAPED, defaults={"equalize": {"taps": 5}})

def test_a_load_time_setting_nothing_reads_warns():
    with pytest.warns(ConfigWarning, match=r"sync 'timing': 'windw' not used by its method"):
        _profile(SHAPED, sync={"timing": {"windw": 1e-3}})

def test_a_method_replaced_at_load_leaves_the_step_s_own_settings_alone():
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConfigWarning)
        _profile(SHAPED + [{"sync": "carrier", "method": "mpower", "order": 4}], sync={"carrier": {"method": "none"}})

def test_every_shipped_profile_reads_all_its_keys():
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConfigWarning)
        for family, modes in rp.profile.ls().items():
            for mode in modes:
                rp.profile.load(f"{family}/{mode}")

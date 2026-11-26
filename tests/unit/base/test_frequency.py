"""Frequency: a float that reads what people write (rpps.base.frequency)"""
import pytest

from rpps.base.frequency import Frequency

@pytest.mark.parametrize("text,hz", [("433.92M", 433.92e6), ("2.4 GHz", 2.4e9), ("10k", 10e3), ("1e6", 1e6), ("50 Hz", 50.0),
                                     ("7h", 7.0), ("-5mhz", -5e6), (".5K", 500.0), (12, 12.0)])
def test_reads_numbers_and_units(text, hz):
    f = Frequency(text)
    assert isinstance(f, float) and f == pytest.approx(hz) and f.raw == pytest.approx(hz)

@pytest.mark.parametrize("text", ["", "fast", "1 THz", "1,5M", "M"])
def test_refuses_what_is_not_a_frequency(text):
    with pytest.raises(ValueError, match="Invalid frequency"):
        Frequency(text)

@pytest.mark.parametrize("hz,shown,mult", [(0, "0.00Hz", 1.0), (999.0, "999.00Hz", 1.0), (1000.0, "1.00KHz", 1e3),
                                           (-5e6, "-5.00MHz", 1e6), (2.4e9, "2.40GHz", 1e9)])
def test_shows_itself_in_its_unit(hz, shown, mult):
    f = Frequency(hz)
    assert str(f) == f.str == shown and f.mult == mult
    assert f.short*f.mult == pytest.approx(hz) and shown.endswith(f.unit)

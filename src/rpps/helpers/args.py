"""argparse argument types"""
from ..base.frequency import Frequency

def arg_freq(f):
    """Process generic frequency arguments"""
    f = Frequency(f)
    f.init()
    return f

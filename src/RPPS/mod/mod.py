import numpy as np

import helpers.binary as binh
from meta import Meta
from .constellation import Constellation

from pyboiler.logger import Logger, Level

Logger().Child("Modulation")

class Modulation:
    constellation = None

    def generate(self):
        ...

    def process(self):
        ...

class ASK(Modulation):
    pass

class FSK(Modulation):
    pass

class APSK(Modulation):
    pass

class QAM(Modulation):
    pass

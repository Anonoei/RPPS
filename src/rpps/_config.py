from pyboiler.logger import Logger, Level
import sys

BYTE_ORDER = ">" if sys.byteorder == "big" else "<"

LOG_CODING = Level.INFO
LOG_CODING_TYPES = Level.WARN

LOG_SCRAM = Level.INFO

LOG_MOD = Level.INFO
LOG_MOD_CONST = Level.WARN

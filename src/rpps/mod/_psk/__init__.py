from .bpsk import BPSK
from .qpsk import QPSK
from .epsk import EPSK

PSK_MAP = {
    BPSK.name: BPSK,
    QPSK.name: QPSK,
    EPSK.name: EPSK,
}

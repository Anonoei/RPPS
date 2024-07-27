from .bpsk import BPSK, BPSK_Maps
from .qpsk import QPSK, QPSK_Maps
from .epsk import EPSK, EPSK_Maps

PSK_MAP = {
    BPSK.name: BPSK,
    QPSK.name: QPSK,
    EPSK.name: EPSK,
}

PSK_MAPPING = {
    BPSK.name: BPSK_Maps,
    QPSK.name: QPSK_Maps,
    EPSK.name: EPSK_Maps,
}

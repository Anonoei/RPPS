from enum import Enum, auto

class Type(Enum):
    NONE = auto()
    DataAided = auto()
    DecisionDirected = auto()
    NonDataAided = auto()

    DA = DataAided
    DD = DecisionDirected
    ND = NonDataAided

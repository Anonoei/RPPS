from enum import Enum, auto
import numpy as np

class Process(Enum):
    NONE = -1
    RAW = auto() # Unprocessed data
    SCRAM = auto() # scrambled data
    CODING = auto() # coding data
    MOD = auto() # modulated data
    DEMOD = auto() # demodulated data
    HARD = auto() # hard data
    SOFT = auto() # soft data

class _Data:
    __slots__ = ("_data")
    process: Process = Process.NONE
    dtype = None

    def __init__(self, data=None):
        if data is None:
            data = np.array([], dtype=self.dtype)
        self._data = data

    def __str__(self):
        return f"{self.name()}:{self.dtype}:{len(self)}"

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self._data)

    def __enter__(self, *args, **kwargs):
        return self._data.__enter__(*args, **kwargs) # type: ignore

    def __exit__(self, *args, **kwargs):
        return self._data.__exit__(*args, **kwargs) # type: ignore

    def __iter__(self, *args, **kwargs):
        return self._data.__iter__(*args, **kwargs)

    def __getitem__(self, *args, **kwargs):
        return self._data.__getitem__(*args, **kwargs)

    @property
    def name(self):
        """Get class name"""
        return type(self).__name__

    def append(self, data):
        self._data = np.append(self.data, data)

class Digital(_Data):
    pass

class Bit(Digital):
    dtype = bool

    def __str__(self):
        return f"{super().__str__()}:{self.bin}"

    @property
    def bin(self):
        return self._data.astype(int)

    def to_bytes(self):
        if isinstance(self._data, np.ndarray):
            return np.packbits(self.data)

    @classmethod
    def from_bytes(cls, data):
        if data is None:
            return cls()
        elif instance(data, Byte):
            return cls(data.to_bits())
        elif isinstance(data, bytes):
            return cls(np.unpackbits(np.array(data)))
        elif isinstance(data, np.ndarray):
            if data.dtype == np.uint8:
                return cls(np.unpackbits(data))
            elif data.dtype == np.bool:
                return cls(data)
            raise NotImplementedError(f"Cannot convert {type(data)} {data.shape} {data.dtype} to {cls.__name__}")
        raise NotImplementedError(f"Cannot convert {type(data)} to {cls.__name__}")

class Byte(Digital):
    dtype = np.uint8

    def __str__(self):
        return f"{super().__str__()}:{self.hex}"

    @property
    def hex(self):
        return self._data.tobytes().hex()

    def to_bits(self):
        if isinstance(self._data, np.ndarray):
            return np.unpackbits(self.data)

    @classmethod
    def from_bits(cls, data):
        if data is None:
            return cls()
        elif instance(data, Bits):
            return cls(data.to_bytes())
        elif isinstance(data, bytes):
            return cls(np.frombuffer(data, dtype=self.dtype))
        elif isinstance(data, np.ndarray):
            if data.dtype == np.uint8:
                return cls(data)
            elif data.dtype == np.bool:
                return cls(np.packbits(data))
            raise NotImplementedError(f"Cannot convert {type(data)} {data.shape} {data.dtype} to {cls.__name__}")
        raise NotImplementedError(f"Cannot convert {type(data)} to {cls.__name__}")

class Analog(_Data):
    pass

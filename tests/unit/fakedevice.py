"""A pretend GPU array module for tests: numpy underneath, but it refuses host/device mixing the way CuPy does (rpps.parallel)

    import fakedevice as xp
    d = xp.asarray(host)          # host -> "device"
    d.get()                       # "device" -> host (rpps.base.kernel.to_host)
    np.asarray(d)                 # TypeError, as CuPy: no implicit conversion
    d + host_array                # TypeError: a numpy array (not a scalar) mixed into device work

A batch kernel that runs here without error moves nothing implicitly, so it runs on CuPy (CUDA or ROCm) as far as array
semantics go. Numbers come out of numpy, so results compare exactly with the numpy run.
"""
import numpy as np

__name__ = "fakedevice"

class DeviceError(RuntimeError):
    pass

def _is_host(v):
    return isinstance(v, np.ndarray) and v.ndim > 0

def _unwrap(v, what):
    if isinstance(v, Dev):
        return v._a
    if _is_host(v):
        raise DeviceError(f"{what}: a numpy array of shape {v.shape} mixed into device work (move it with xp.asarray)")
    if isinstance(v, (list, tuple)):
        return type(v)(_unwrap(x, what) for x in v)
    if isinstance(v, slice):
        return slice(_unwrap(v.start, what), _unwrap(v.stop, what), _unwrap(v.step, what))
    return v

def _wrap(v):
    if isinstance(v, np.ndarray):
        return Dev(v)
    if isinstance(v, tuple):
        return tuple(_wrap(x) for x in v)
    if isinstance(v, list):
        return [_wrap(x) for x in v]
    return v

class Dev(np.lib.mixins.NDArrayOperatorsMixin):
    """An array on the pretend device"""
    __slots__ = ("_a",)

    def __init__(self, a):
        object.__setattr__(self, "_a", a)

    def __array__(self, *a, **k):
        raise DeviceError("implicit conversion of a device array to numpy (use .get(), rpps.base.kernel.to_host)")

    def __array_ufunc__(self, ufunc, method, *inputs, **kw):
        ins = _unwrap(inputs, ufunc.__name__)
        if "out" in kw:
            kw["out"] = _unwrap(kw["out"], ufunc.__name__)
        return _wrap(getattr(ufunc, method)(*ins, **kw))

    def __array_function__(self, func, types, args, kw):
        return _wrap(func(*_unwrap(args, func.__name__), **_unwrap(kw, func.__name__)))

    def get(self):
        return self._a.copy()

    def __getitem__(self, i):
        return _wrap(self._a[_unwrap(i, "index")])

    def __setitem__(self, i, v):
        self._a[_unwrap(i, "index")] = _unwrap(v, "assignment")

    def __getattr__(self, name):
        if name.startswith("__"): # (__array_interface__, the buffer protocol: numpy's ways in, which CuPy does not give)
            raise AttributeError(name)
        v = getattr(self._a, name)
        if callable(v):
            return lambda *a, **k: _wrap(v(*_unwrap(a, name), **_unwrap(k, name)))
        return _wrap(v)

    def __len__(self):
        return len(self._a)

    def __iter__(self):
        return (_wrap(x) for x in self._a)

    def __bool__(self):
        return bool(self._a)
    def __float__(self):
        return float(self._a)
    def __int__(self):
        return int(self._a)
    def __index__(self):
        return self._a.__index__()
    def __complex__(self):
        return complex(self._a)

    def __repr__(self):
        return f"Dev({self._a!r})"

def asarray(v, dtype=None):
    """Host or device in, device out (CuPy's asarray takes both)"""
    if isinstance(v, Dev):
        return Dev(v._a if dtype is None else v._a.astype(dtype, copy=False))
    return Dev(np.array(v, dtype=dtype))

array = asarray

def asnumpy(v):
    return v.get() if isinstance(v, Dev) else np.asarray(v)

def __getattr__(name):
    v = getattr(np, name)
    if callable(v) and not isinstance(v, type):
        return lambda *a, **k: _wrap(v(*_unwrap(a, name), **_unwrap(k, name)))
    return v

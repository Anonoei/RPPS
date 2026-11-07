from ..base import Meta, Stage

def Sync(impl):
    """Kept for older code: synchronizers are stages now, so `block + Costas(...)` works directly"""
    return impl

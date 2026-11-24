"""Code types: the FEC encoders and decoders the definitions in config/fec and config/interleave build

A code type subclasses `rpps.code.types._code._code` and defines `_decode(x, ...) -> (output, errors per codeword or
None)`, keeping nothing of the call on itself, so one loaded (cached) code decodes on several threads at once. Soft or
path outputs are methods that return them too (`app` of BCJR and turbo, `paths` of fsm), never attributes.

## A trellis, worked

The rate 1/2, constraint length 3 code with generators 101 and 111 (state = the last two input bits):

| State | In | Out | Next |
|---|---|---|---|
| 00 | 0 | 00 | 00 |
|    | 1 | 11 | 10 |
| 01 | 0 | 11 | 00 |
|    | 1 | 00 | 10 |
| 10 | 0 | 01 | 01 |
|    | 1 | 10 | 11 |
| 11 | 0 | 10 | 01 |
|    | 1 | 01 | 11 |

```mermaid
graph LR
    S00((00)) -- 0/00 --> T00((00))
    S00 -- 1/11 --> T10((10))
    S01((01)) -- 0/11 --> T00
    S01 -- 1/00 --> T10
    S10((10)) -- 0/01 --> T01((01))
    S10 -- 1/10 --> T11((11))
    S11((11)) -- 0/10 --> T01
    S11 -- 1/01 --> T11
```

`fsm` holds such tables for any code (`output[s][u]`, `next[s][u]`); `viterbi` and `bcjr` decode on them.
"""
from ._code import REGISTRY, register
from .bchgf import bchgf
from .ira import ira_checks, ira_encode
from .conv import conv
from .interleave import forney, block_interleaver, rowcol
from .fsm import fsm
from .turbo import turbo, rsc, qpp
from .ratematch import ratematch
from .nr_polar import nr_polar
from .nr_ldpc import nr_ldpc
from .product import product
from .ldpc import ldpc, gallager
from .linear import linear
from .expr import expr
from .layout import arranged
from .cyclic import cyclic
from .syndrome import syndrome
from .repeat import repeat
from .rs import rs
from .polar import polar
from .rm import rm
from .viterbi import viterbi
from .trellis import Trellis, bcjr, bcjr_conv, trellis

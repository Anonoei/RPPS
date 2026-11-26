# The data model

RPPS has no graph to build and nothing to reset. Operators are applied to an `rp.Data`, which knows its representation
(samples, symbols, channel bits, data bits, packets), its rates, and the state of every operator that has touched it.

| Operator | Means | On |
|---|---|---|
| `data * pipe`, `data / pipe` | encode / decode | `Pipe`: an invertible transform (symbol, code, link) |
| `data + stage` | process | `Stage`: DSP on a stream (signal, sync, wave) |
| `pipe * pipe`, `stage + stage` | a reusable `Pipeline` | either |
| `data >> sink` | write out | `Sink` |

The design behind it (the chain samples → symbols → channel bits → data bits → packets, which package each function
belongs in, and what each category may take and give) is in [`rpps.base`](../api/base.md).

## Bits to symbols and back

```python
--8<-- "examples/quickstart.py"
```

## Things that surprise people

- **What `rp.Data` makes of an array.** `bytes` and `uint8` arrays are bytes (`[0, 1, 1, 0]` as uint8 is 32 bits); bits
  are `bool` (an int 0/1 array is an error, not bits); a float array is real samples, never LLRs; `complex64` with
  `sps=1` is symbols, `complex128` stays samples. `Fs` defaults to 1.0.
- **Partial blocks wait.** A code or constellation keeps what doesn't fill a block in the data's context until more
  comes or the block is flushed: three bits through QPSK give one symbol, or two with `rp.Data(bits, flush=True)`.
- **Two kinds of carried state.** A `Stage` (`+`) keeps its state in the stream: pass `stream=prev.stream` to go on.
  A `Pipe` (`*`, `/`) keeps it in the context: `ctx=prev.ctx`. A fresh `rp.Data(chunk)` starts both over. Iterating a
  source does this for you.
- **Some state lives on the operator.** `Feedthrough.encode(array)` called directly moves the register (`reset()` starts
  it over); `control()` changes the operator for everyone holding it. Decode status is not kept on the operator: it
  comes with the output, `out.errors` after `data / code` (per codeword: corrections, -1 where it failed) or
  `msg, errors = code.decoder.decode(x, status=True)`, so threads can share one loaded code.
- **Bits to a link layer go through `rp.code.Uncoded()`.** `syms / qpsk / crc` is an error: channel bits are not data
  bits until a code (or `Uncoded`) says so.
- **LLRs are log(P0/P1)**: positive means 0, a hard decision is `llr < 0`.
- **Bit order is most significant first** (bytes, constellation labels, RS symbols, markers), except HDLC bytes on the
  wire (least significant first), recurrence `init` integers (`lsb` unless `"order": "msb"`) and headers that say `"lsb"`.
- **Shapes and definitions are separate.** `rp.symbol.load("psk/qpsk")` reads a definition; `rp.symbol.generate("16QAM")`
  makes a shape; DVB-S2's 16APSK is `rp.symbol.load("apsk/dvbs2_16apsk", rate="3/4")`, not `generate("16APSK")`.
- **Shaping needs a symbol rate.** `rp.Data(bits, Fs=1e6, sps=1)` before a pulse: a pulse is designed from the stream's
  samples per symbol.

## Operators, links and partial blocks

```python
--8<-- "examples/operators.py"
```

## Soft decisions

`symbols / constellation` gives LLRs, and every decoder that can use them does.

```python
--8<-- "examples/soft_decisions.py"
```

More recipes (DSSS, O-QPSK, 802.15.4, GFSK) are in [`rpps.symbol`](../api/symbol.md).

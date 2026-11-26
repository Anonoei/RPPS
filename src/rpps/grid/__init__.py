"""Resource grids of OFDM systems: cell search, the grid of each subframe with its channel estimates, and the physical
channels read off it

`grid` steps in a profile take an OFDM system's samples to frames of its resource grid, and its physical channels read
them, a branch each (`rpps.profile.branch`). Formula mappings are code here; tables and sequences are definitions.

- `lte` (FDD and TDD): `LteCell` (PSS/SSS cell search; `search` lists the cells on a carrier), `LteGrid` (a frame per
  subframe, with each port's channel), `PBCH` (ports, MIB), `PCFICH` (CFI), `PDCCH` (DCI 1A, 1C), `PDSCH` (SIB1, SI),
  and `tx`. The profile is `lte/downlink`.
- `nr` (FR1, 15 and 30 kHz): `NrCell` (PSS over a LEO pass's offsets, each SS/PBCH block tracked), `NrGrid` (a frame per
  subframe), `PBCH`, `PDCCH` (CORESET 0), `PDSCH` (LDPC, SIB1), and `tx` (with `leo`). Not every cell applies the per-symbol
  phase of 38.211 5.4: with `fc` the PDSCH tries both. The profile is `nr/downlink`.
"""
from . import lte

"""Resource grids of OFDM systems: cell search, the grid of each subframe with its channel estimates, and the physical
channels read off it

`grid` steps in a profile take an OFDM system's samples to frames of its resource grid, and its physical channels read
them, a branch each (`rpps.profile.branch`). Each channel publishes what it decoded as a header does, and the controls
carry it to the channels after it. Mappings the standard gives by formula (where the reference signals, the REGs and the
allocation are) are code here; tables (transport block sizes) and sequences (the Gold sequence) are definitions.

- `lte`: `LteCell` (PSS/SSS cell search: the cell id, the carrier offset; half-frames from a subframe 0, sample clock
  tracked), `LteGrid` (a frame per subframe, its grid and the channel of ports 0-3 from the reference signals), the
  channels `PBCH` (port count by its CRC mask, the MIB), `PCFICH` (the CFI), `PDCCH` (a DCI 1A to the SI-RNTI), `PDSCH`
  (the transport blocks: SIB1, the SI messages), and `tx` (a downlink to test with). The profile is `lte/downlink`.
- `nr` (15 kHz): `NrCell` (the PSS over a wide offset: a LEO pass at S-band moves the carrier by some 50 kHz; each SS/PBCH
  block's timing and offset tracked with their rates), `NrGrid` (a frame per slot), `PBCH` (MIB, SFN), `PDCCH`
  (CORESET 0 and the Type0 search space from the MIB; DCI 1_0, polar), `PDSCH` (LDPC, SIB1), `tx` (a downlink, and
  `leo`: a satellite pass). NR's channels each estimate the channel from their own DM-RS. The per-symbol phase of NR's
  upconversion (38.211 5.4) is e^(-j 2 pi fc t_l) on symbol l; not every cell applies it, so with `fc` the cell search
  tells which and the PDSCH tries both, its CRC choosing. The profile is `nr/downlink`.
"""
from . import lte

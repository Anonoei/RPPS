# Working changes

Local notes on what is in progress on `dev` (untracked: not committed). Finished work goes to CHANGELOG.md (Unreleased)
as it lands. Updated 2026-11-25.

## Since 0.6.0

Released work is in CHANGELOG.md. This cycle: sync by block estimation (loops second class in `sync.loops`), the
level-based finder (`find_signals`, `find`, streaming `Finder`), burst analysis as `rpps.analysis.burst` with time plans
(`rpps.plan`, MF-TDMA, DVB-RCS2), P25 / DMR profiles, codes bit by bit, the GPU region, blind identification work.

## Release 0.7.0

- [x] CHANGELOG Unreleased consolidated to net changes since 0.6.0
- [x] Roadmap brought up to date
- [ ] `python tools/builder.py -d` (strict docs build)
- [ ] Slow tests; kernels rebuilt for each Python shipped
- [ ] `python tools/builder.py -v m`, then `-r`; merge to `main`; GitHub release `v0.7.0`
- After 0.7.0: remove `rp.analysis.spectrum.burst` and `find_signals`' retired arguments (both warn this release)

## Open

- **Finder:** profile `_refit` on the level reading; a `Finder` in pyspecan.
- **GPU:** MPower and Oerder & Meyr on the device wait on it every window (1.3x slower end to end than the host on the
  860M): measure a group of windows a launch, as MatTrack does.
- **Speed:** `lte/downlink` pipelined is 0.95x real time at 15.36 MS/s in 262144-sample blocks, 1.0x in 65536 (one
  thread 1.01x and 1.1x): the GIL holds the trunk and branches back; the branches (PDCCH blind decodes, per-subframe
  Python) are the next to trim.
- **NR:** 30 kHz SS/PBCH or CORESET; more than 4 candidate blocks; VRB-to-PRB interleaving; several code blocks a
  transport block; co-channel cells.
- **LTE past SIB1:** unicast DCI formats, several code blocks, TDD, extended prefix.
- **Transmit side of branch profiles** (`grid.lte.tx` and `grid.nr.tx` are test transmitters).
- **Codes:** NR polar parity-check bits (UL); LDPC limited-buffer rate matching and HARQ combining.

The longer list is the roadmap's [Not yet built](docs/roadmap.md#not-yet-built).

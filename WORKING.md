# Working changes

What is in progress on `dev`, not yet released. Finished work moves to [CHANGELOG.md](CHANGELOG.md) (Unreleased) as it
lands; this file says what is being built now, what is next and what is open. Updated 2026-11-22.

## The last goal: done

Burst and preamble steps in profiles, then LTE and 5G NR (NTN) downlinks through SIB1, with branches in one profile.
Every phase is in CHANGELOG.md (Unreleased):

| Phase | What | Commit |
|---|---|---|
| 1 | Burst demodulation in profiles: `{"sync": "burst"}`, `{"sync": "preamble"}`, known timing anchored; `ble/adv` per packet | e975d84 |
| 2 | Branches in profiles: one trunk, chains reading the same frames, headers driving across | 1dfbf42 |
| 3 | `rpps.grid`, `lte/downlink` to SIB1 (band 13 capture: MIB 100/100, SI blocks 60/60; live: 100/100, 58/58) | 4046b86 |
| 4 | NR polar and LDPC with their rate matching; checked against py3gpp | 8eb71fc, 26275ae |
| 5 | `nr/downlink` to SIB1 at 15 kHz, through a LEO pass; live cells in n5 (SIB1 41/42) and n25 (50/50) | |
| DVB | DVB-S2 multistream in branches; trunk controls drive branch steps; header select | ebe1694 |

Since then: burst transmission in profiles, decoded `Message` records, `profile.Decoder`, message kinds, `sync.Lock`,
and the effort levels measured.

## Checked independently, and not

py3gpp (a scratch venv only, not a dependency) gave the vectors in `tests/unit/vectors/nr_py3gpp*.json`:

- **Bit-exact:** LDPC codewords and rate matching (BG1 and BG2, redundancy versions, filler, the QPSK and 16QAM
  interleaver); all six CRCs; polar where py3gpp implements NR (repetition, E >= N); the SS/PBCH block, BCH and PBCH
  scrambling.
- **Not covered:** py3gpp's polar always takes N = 2^nmax and freezes nothing for rate matching. The N rule and the bits
  frozen for puncturing or shortening follow the 38.212 text, with nothing independent to check them against.

## Open

- **Speed:** `lte/downlink` is about 1.1x real time at 15.36 MS/s: the FFTs, channel estimation and the PDCCH's many
  small Viterbi decodes are what is left; running the trunk and branches on separate cores (`pipelined()`) is next.
- **NR:** 30 kHz SS/PBCH or CORESET (Tables 13-2 to 13-6); more than 4 candidate blocks (above 3 GHz); VRB-to-PRB
  interleaving; several code blocks a transport block; co-channel cells (a single-cell receiver only).
- **LTE past SIB1:** unicast DCI formats, several code blocks, TDD, extended prefix.
- **Transmit side of branch profiles** (`grid.lte.tx` and `grid.nr.tx` are test transmitters).
- **Codes:** NR polar parity-check bits (UL); LDPC limited-buffer rate matching and HARQ combining.

The longer list is the roadmap's [Not yet built](docs/roadmap.md#not-yet-built).

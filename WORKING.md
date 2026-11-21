# Working changes

What is in progress on `dev`, not yet released. Finished work moves to [CHANGELOG.md](CHANGELOG.md) (Unreleased) as it
lands; this file says what is being built now, what is next and what is open. Updated 2026-11-21.

## The goal

Burst and preamble steps in profiles, so burst demodulators can be profiles, then LTE and 5G NR NTN support. Scope agreed:

- **LTE:** downlink receive through SIB1.
- **5G NR:**
  - NTN, LEO: large Doppler and Doppler rate.
  - Downlink through SIB1: SSB, PBCH, CORESET0 and PDCCH, PDSCH, DMRS, NR polar and LDPC.
- **Grid design:** branches in one profile. One profile whose resource grid fans out to a branch per physical channel,
  where what one channel decodes drives the channels after it.

## Done (in CHANGELOG.md, Unreleased)

| Phase | What | Commit |
|---|---|---|
| 1 | Burst demodulation in profiles: `{"sync": "burst"}`, `{"sync": "preamble"}`, known timing anchored; `ble/adv` per packet | e975d84 |
| 1 | Burst floor from the noise windows alone (bursts may fill nearly all the air) | 57a819b |
| 2 | Branches in profiles: one trunk, chains reading the same frames, headers driving across | 1dfbf42 |
| 3 | `rpps.grid`, `lte/downlink` to SIB1 (band 13 capture: MIB 100/100, SI blocks 60/60; live: 100/100, 58/58) | 4046b86 |
| 4 | NR polar and LDPC with their rate matching; checked against py3gpp | 8eb71fc, 26275ae |
| DVB | DVB-S2 multistream in branches; trunk controls drive branch steps; header select | ebe1694 |

## In progress

- **NR codes against an independent implementation.** py3gpp, in a scratch venv only (not a dependency), gave vectors
  now in `tests/unit/vectors/nr_py3gpp.json`:
  - **Bit-exact:** LDPC codewords and rate matching (BG1 and BG2, redundancy versions, filler, the QPSK and 16QAM
    interleaver); all six CRCs; polar where py3gpp implements NR (repetition, E >= N).
  - **Not covered:** py3gpp's polar always takes N = 2^nmax and freezes nothing for rate matching. Against that
    simplification the polar sequence, input interleaver and encoder match bit for bit. The N rule and the bits frozen
    for puncturing or shortening follow the 38.212 text, with nothing independent to check them against.
- **DVB and branches: done.**
  - DVB-S2 multistream (`dvbs2/mis`): a branch per input stream, each BBHEADER keeping its ISI (`select`).
  - Controls in the trunk now drive steps in branches; here the PLS sets every stream's BBFRAME length.
  - DVB-S2's PLS-to-payload control stays in one chain: it is forward, and its frames change length.
  - DVB-S carries one transport stream: nothing to branch.

## Next

- **Phase 5, NR through SIB1 with NTN: built** (`grid.nr`, `nr/downlink`):
  - synthetic: terrestrial, and a LEO pass at 2.185 GHz from 300 s out (50 kHz) through closest approach (over 600 Hz/s);
  - the SS/PBCH block, BCH and PBCH scrambling are bit-exact against py3gpp.
- **Next: a real NR signal.** A live NR cell at 15 kHz (n71, 617-652 MHz) is in the HackRF's range. Its PBCH passing
  CRC-24C would check the chain end to end, past the synthetic tests and py3gpp.
- **NR scope left:**
  - 30 kHz SS/PBCH or CORESET (Tables 13-2 to 13-6);
  - more than 4 candidate blocks (above 3 GHz);
  - VRB-to-PRB interleaving;
  - several code blocks a transport block;
  - real time (about 7x slower at 7.68 MS/s).

## Open

- `lte_grid` is about 5x slower than real time at 15.36 MS/s: `LteCell`'s derotation and the CRS interpolation.
- Not built:
  - the transmit side of branch profiles (`grid.lte.tx` is a test transmitter);
  - LTE past SIB1: unicast DCI formats, several code blocks, TDD, extended prefix;
  - NR polar parity-check bits (UL);
  - LDPC limited-buffer rate matching and HARQ combining.

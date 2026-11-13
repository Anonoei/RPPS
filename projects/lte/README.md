# lte: LTE downlink decoding

Run from the repo root. [LTE.md](LTE.md) has the method, every result (cells, SIB contents, traffic), the traps and the RPPS-promotion list.
Order of the chain: `lte_cell` (cell search) -> `lte_pbch` (MIB) -> `lte_sib` (PCFICH/PDCCH/PDSCH -> SIB1, SI messages, with `lte_turbo` and `lte_rrc`) -> `lte_scan` (scan, identify, traffic).

## lte_cell.py: cell search

    python projects/lte/lte_cell.py data/lte_b13.cf64 --fs 15.36 --cf 753 --carrier 751.0

PSS over a frequency grid, per-occurrence timing (and the sample-clock error), cyclic-prefix offset, then SSS for the cell ID and half-frame.
**projects/lte/LTE.md** has the method, what went wrong along the way, and the plan.

## lte_pbch.py: PBCH -> MIB

    python projects/lte/lte_pbch.py data/lte_b13.cf64 --fs 15.36 --cf 753 --carrier 751.0
    python projects/lte/lte_pbch.py --selftest

Runs the cell search, then channel estimate from the CRS, 1/2/4-port equalization, descrambling, rate dematching, tail-biting Viterbi and CRC. Band 13: 50 of 50 frames decode,
4 antenna ports, 50 RB, system frame number counting 551..600, LTE-M scheduled. The sub-functions (`gold`, `crs_values`, `conv_encode`, `viterbi_tailbiting`, `crc16`, `parse_mib`) are written to be lifted into RPPS.

## lte_sib.py, lte_turbo.py, lte_rrc.py: SIB1

    python projects/lte/lte_sib.py data/lte_b13.cf64 --fs 15.36 --cf 753 --carrier 751.0
    python projects/lte/lte_turbo.py --selftest
    python projects/lte/lte_rrc.py build 36331-j30.docx eutra.asn && python projects/lte/lte_rrc.py decode eutra.asn <hex>   # needs asn1tools in a venv

PCFICH, PDCCH blind decode for the SI-RNTI, DCI 1A, PDSCH with 4-port transmit diversity, a turbo codec (QPP table from the spec, max-log-MAP) and the 36.331 ASN.1 compiled from the
spec itself. Band 13: SIB1 decoded 25/25 (PLMN 311-480, TAC 0x0701, cell identity 0x01D1102 (eNodeB 7441, cell 2), band 13); with `--asn` it also decodes SIB2/3/4/5/8 from the SI windows (SIB5 names two more carriers: 885 and 2120 MHz). See LTE.md.

## lte_scan.py: scan, identify, traffic

    python projects/lte/lte_scan.py --scan --traffic --keep captures/          # sweep default bands, identify every cell, measure the load
    python projects/lte/lte_scan.py --carrier 885.0 --bw 10 --traffic          # one known carrier
    python projects/lte/lte_scan.py --file cap.cf64 --fs 15.36 --cf 887 --carrier-nominal 885.0 --traffic

Sweep -> LTE-shaped plateaus -> gain search -> capture (resampled from 20 MS/s to the LTE rate) -> PSS/SSS/MIB/SIB1 -> PRB utilisation, CFI and public-RNTI grant counts. A validity gate rejects PSS
correlations that are only noise. Aggregates only: no user data, no paging records. See LTE.md.

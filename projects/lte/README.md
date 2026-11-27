# lte: LTE downlink decoding

Run from the repo root. [LTE.md](LTE.md) has the method, every result (cells, SIB contents, traffic), the traps and the RPPS-promotion list.
`lte_sib` (cells on a carrier -> SIB1 and SI messages, with `lte_rrc`) and `lte_scan` (scan, identify, traffic) use the library receiver, `rpps.grid.lte`. `lte_cell` (cell search) and `lte_pbch` (MIB) are the same first steps written out on their own, with their statistics.

## lte_cell.py: cell search

    python projects/lte/lte_cell.py data/lte_b13.cf64 --fs 15.36 --cf 753 --carrier 751.0

PSS over a frequency grid, per-occurrence timing (and the sample-clock error), cyclic-prefix offset, then SSS for the cell ID and half-frame.
**projects/lte/LTE.md** has the method, what went wrong along the way, and the plan.

## lte_pbch.py: PBCH -> MIB

    python projects/lte/lte_pbch.py data/lte_b13.cf64 --fs 15.36 --cf 753 --carrier 751.0
    python projects/lte/lte_pbch.py --selftest

Runs the cell search, then channel estimate from the CRS, 1/2/4-port equalization, descrambling, rate dematching, tail-biting Viterbi and CRC. Band 13: 50 of 50 frames decode,
4 antenna ports, 50 RB, system frame number counting 551..600, LTE-M scheduled. The sub-functions (`gold`, `crs_values`, `conv_encode`, `viterbi_tailbiting`, `crc16`, `parse_mib`) are written to be lifted into RPPS.

## lte_sib.py: SIB1 and the SI messages of every cell on a carrier

    python projects/lte/lte_sib.py data/lte_b13.cf64 --fs 15.36 --cf 753 --carrier 751.0
    python projects/lte/lte_sib.py cap.cf64 --fs 30.72 --cf 2120 --carrier 2120 --pci 25

A front end to `rpps.grid.lte` (the profile `lte/downlink`): the cell search lists the cells on the carrier, and each is received: MIB, PCFICH, PDCCH blind decode for the SI-RNTI
(DCI 1A, localized or distributed, and 1C), PDSCH with 1, 2 or 4 antenna ports, FDD or TDD. With `--asn` (see below) it decodes the messages with the 36.331 ASN.1 and names each
SI message by the window SIB1 schedules it in. Band 13: SIB1 decoded 50/50 (PLMN 311-480, TAC 0x0701, cell identity 0x01D1102 (eNodeB 7441, cell 2), band 13). Not received: PDSCH in
TDD's special subframe.

## lte_turbo.py, lte_rrc.py: the turbo code and the ASN.1

    python projects/lte/lte_turbo.py --selftest
    python projects/lte/lte_rrc.py build 36331-j30.docx eutra.asn && python projects/lte/lte_rrc.py decode eutra.asn <hex>   # needs asn1tools in a venv

## lte_scan.py: scan, identify, traffic

    python projects/lte/lte_scan.py --scan --traffic --keep captures/          # sweep default bands, identify every cell, measure the load
    python projects/lte/lte_scan.py --carrier 885.0 --bw 10 --traffic          # one known carrier
    python projects/lte/lte_scan.py --file cap.cf64 --fs 15.36 --cf 887 --carrier-nominal 885.0 --traffic

Sweep -> LTE-shaped plateaus -> gain search -> capture (resampled from 20 MS/s to the LTE rate) -> PSS/SSS/MIB/SIB1 -> PRB utilisation, CFI and public-RNTI grant counts. A validity gate rejects PSS
correlations that are only noise. Aggregates only: no user data, no paging records. See LTE.md.

# spectrum: wideband sweeps, channel occupancy and gain tuning

Run from the repo root. Needs a supported SDR. Full description in the `wide_survey.py` docstring.

    python projects/spectrum/wide_survey.py sweep data/a                 # two passes (tile grid shifted 8 MHz): data/a_1.npz, data/a_2.npz (~3.5 min)
    python projects/spectrum/wide_survey.py sweep data/b --lo 700 --hi 900
    python projects/spectrum/wide_survey.py report data/a                # confirmed emitters grouped by allocation
    python projects/spectrum/wide_survey.py report data/b --compare data/a

An emitter counts only where both passes show it (rejects the radio's 20 MHz spurs and tile edges). For narrowband channel occupancy and AM/FM in 108-470 MHz use `vhf_scan.py` below.

## sweep_monitor.py: live sweep in the browser

    python projects/spectrum/sweep_monitor.py --lo 30 --hi 2000 --remote tcp://HOST:5555     # then open http://127.0.0.1:8026/

Sweeps continuously and shows the stitched spectrum with the tile the radio is on, the sweep in progress, the average of the last 8 sweeps and the peak hold, a spectrogram of the completed sweeps
(newest on top), the allocation bands, and the strongest signals in the average. Drag on the spectrum to zoom; the range, pause and scale are on the page, and so are the sweep settings: tile width (1-20 MHz), sample rate (auto = tile / 0.8; limited to the radio's rates from pysdrlib's caps), dwell per tile in samples or seconds, settle time after each retune, gain (auto ladder or fixed amp / LNA / VGA) and whether the tile grid shifts every other sweep (the same as `--tile`, `--fs`, `--dwell-samples`, `--dwell-seconds`, `--settle`, `--gain` on the command line). A narrower tile keeps away from the filter edges at the cost of more tiles; a longer dwell averages the noise down for weak or bursty signals. A 500 MHz range takes about 9 s per sweep,
30-2000 MHz about 35-60 s. The radio is used exclusively, so stop other clients first. 

### Keeping artifacts and floor steps out of it

Things the monitor does (each has a switch on the page and on the command line; `--debug-tiles` prints per tile the gain, floor, edge-versus-centre level, whether the shape was subtracted and the level match):

- **Held gain** (`--headroom`, default 0.5 of full scale): a tile starts at the previous tile's gain and steps down only when its peak amplitude passes the headroom, with one step up tried every 12 tiles. Choosing by clipping alone let intermodulation through, and a different gain per tile gave steps in the floor.
- **Flattened tiles** (`--no-flatten`): the median over recent tiles of (tile minus its own middle level), per offset from the tile centre, is subtracted. It holds what every tile shares (the roll-off towards the tile edges, spurs at a fixed offset from the centre); a real signal is at a different offset in each tile and does not enter the median. It needs 8 tiles per sweep, and a range of only a few tiles never gets a shape. The whole sweep is redone with the final shape when it ends.
- **Gain-setting errors** (`--no-match`): the nominal gain correction (amp + LNA + VGA) is not what the radio does, so the floor steps where the setting changes. At each join between tiles measured at different settings the difference over the strip both measured is an observation; the error of each setting is the least-squares solution over the joins seen (the most used setting is the reference; a setting no join reaches keeps the nominal level). Joins at the same setting are left alone, so a real slope of the floor along the band survives. (The first version chained neighbouring tiles along the sweep: it flattened the real slope, hit its 8 dB limit and let the steps back.)
- **Confirmation** (`--no-alt-fs`, `--no-alt-gain`, `--peak-decay`): odd sweeps use 80 % of the sample rate (20 -> 16 MS/s, tile 16 -> 12.8 MHz), the half-tile shifted grid and half a rung lower gain (`LADDER_ALT`, highest gain with headroom). The first two move combs and images; overload and intermodulation products of the front end stay at the same frequencies whatever the tuning or rate, and only less gain removes them (a third-order product falls twice as fast as the gain, input-referred). The average, peak hold, spectrogram and signal list take a sweep only where both neighbouring sweeps agree within 4 dB, else the lowest value (the newest sweep is revised when the next one finishes, and the peak hold lags one sweep); combs and images move with the sweep geometry, signals do not. What only one sweep showed is drawn in red; a burst seen once is there, not in the average. The peak hold falls 0.25 dB per sweep and has a reset button.

## vhf_scan.py: VHF / UHF channel scanner (108-470 MHz)

Needs a supported SDR.

    python projects/spectrum/vhf_scan.py                                   # 108-470 MHz, 3 s per 16 MHz tile
    python projects/spectrum/vhf_scan.py --lo 118 --hi 137 --seconds 10    # aviation voice band, longer
    python projects/spectrum/vhf_scan.py --keep data/vhf/                  # keep each tile's capture (vhf_<MHz>.cf64)
    python projects/spectrum/vhf_scan.py --files "data/vhf/vhf_{}.cf64" --lo 144 --hi 160 --cf 152   # analyse a saved tile

Per tile: highest non-clipping gain (`radio_io.capture_clean`), 12.5 kHz channel bins, 10 ms frames; per channel the duty (share of active frames), level, width and an AM-vs-constant-envelope test.
Channels are labelled from the North American allocation and channel tables in the file (what the frequency is *for*, not what was decoded).
For the details of one signal (width, baud), analyse a kept tile with `python -m rpps survey data/vhf/vhf_<MHz>.cf64 --fs 20e6 --fmt cf64 --cf <Hz>`.

## Analysis tools (work on a saved cf64 capture, or capture live)

    python projects/spectrum/channel_watch.py data/avi/avi_125.cf64 --fs 8 --cf 125            # which channels are active, duty, likely radio spurs
    python projects/spectrum/channel_watch.py --tune 407 --fs 4 --seconds 20 --gain 0.85 --keep data/fed/fed_407.cf64
    python projects/spectrum/burst_tones.py data/fed/fed_142.4.cf64 --fs 2 --cf 142.4 --freq 142.8525 --list 60   # bursts, FM stats, tone per burst
    python projects/spectrum/fm_audio.py data/lmr/lmr_462.5.cf64 --fs 8 --cf 462.5 --freq 464.8062 --wav out.wav   # FM audio of a channel + speech / tone check
    python projects/spectrum/voice_scan.py --wav-dir data/voice                                           # scan bands for voice: tunes the gain per band, labels P25 / DMR / FM / AM / carrier

    python projects/spectrum/fsk_probe.py data/lmr/p25_854.cf64 --fs 8 --cf 854 --freq 851.425 853.325   # FSK or not, levels, baud-rate line (4800 P25 1 / DMR, 6000 P25 2, 9600 ...)

`fsk_probe.py` is a screening tool, not a decoder: its baud-rate line is not always the strongest on a known channel (it picked 2987 Hz on a C4FM control channel), so use it to rank channels, then confirm.

Tune the gain first with `gain_search.py` (keep the signal band away from the tuner centre: the DC spike). Lines that stay put when the sample rate and centre change are real; combs that
move with the sample rate are artifacts. These are intended to move into RPPS.

"""What the reference tools share: a capture from a file or the radio, with its receive context"""


def add_source_args(ap, fs=15.36, seconds=0.6, gain=0.45, freq_required=True):
    """The arguments `source` reads: --freq, --fs, --file, --fmt, --seconds, --gain"""
    ap.add_argument("--freq", type=float, required=freq_required, help="centre frequency over the air [MHz]")
    ap.add_argument("--fs", type=float, default=fs, help=f"sample rate [MS/s] (default {fs:g})")
    ap.add_argument("--file", help="a capture instead of the radio (a .sigmf brings its receive context)")
    ap.add_argument("--fmt", default="cf32", help="the capture's pysdrlib format (default cf32; not for .sigmf)")
    ap.add_argument("--seconds", type=float, default=seconds, help=f"seconds to capture (default {seconds:g})")
    ap.add_argument("--gain", type=float, default=gain, help=f"receive gain, a fraction of the radio's range (default {gain:g})")


def source(a, file=None):
    """A source with `peek` from the parsed arguments: the file (`file` overrides --file), else `seconds` of the radio
    (radio_io: $RPPS_RADIO, PYSDRLIB_REMOTE) captured into memory with its receive context"""
    import rpps as rp
    Fs, cf = a.fs*1e6, (a.freq or 0.0)*1e6
    path = file or a.file
    if path:
        if str(path).endswith((".sigmf", ".sigmf-meta", ".sigmf-data")):
            return rp.source.SigMF(path)
        return rp.source.File(a.fmt, path, Fs=Fs, cf=cf)
    import radio_io
    dev = radio_io.device()
    try:
        live = rp.source.Device(dev, cf=cf, Fs=Fs, rx_gain=a.gain)
        return rp.source.Memory.capture(live, int(a.seconds*Fs), skip=int(0.45*Fs)) # (a retuned radio delivers the old frequency a while)
    finally:
        dev.close()

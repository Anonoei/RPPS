"""A catalogue of radio systems by what the blind tools measure: features in, named systems out, with the reasons

    f = rp.analysis.catalogue.features(fsk=r, frequency=935.2e6)       # from tool results (or a dict by hand)
    for m in rp.analysis.catalogue.match(f)[:3]:
        print(m)        # "GSM (3GPP TS 45.004/45.002): 0.93 - modulation gmsk, baud 270833 ~ 270833, frame ..."

Features (any subset; a tool result fills the ones it measures, `features` maps them):
  modulation   a family name: gmsk msk gfsk 2fsk 4fsk bpsk qpsk oqpsk 8psk pi4dqpsk 16qam 16apsk 32apsk soqpsk ofdm ppm
               css 8vsb pm fm am (rpps.analysis tools' own names are mapped: Detection.mod, FSK.name, cpm schemes)
  baud         symbol rate (Hz); chip_rate and code_length for spread signals
  bt, h        a Gaussian pulse's bandwidth-time, a CPM / FSK index
  beta         a root-raised-cosine roll-off
  scs, cp      OFDM subcarrier spacing (Hz) and cyclic prefix (s); pilot_spacing, pilot_step, pilot_period (symbols)
  frame, slots a TDMA / TDD frame (s) and its slots
  bandwidth    occupied (Hz); frequency (Hz: the carrier, a weak hint: allocations differ by region)

Each system lists values (a number, a list of allowed numbers, a (lo, hi) range, or a set of names) with a relative
tolerance. A feature both sides have scores +weight when it agrees and -weight when it contradicts; the score is the
sum over the most it could have been plus `prior` (so a fit on one feature ranks below a fit on several: 1 is
approached as more agrees; below 0 more contradicts than agrees), and the reasons say which. Nothing measured that a system lacks counts either way. A blind tool reads some modulations as
others (pi/4-DQPSK as 8PSK, GMSK as MSK or 2-FSK, C4FM as 4-FSK): systems list what they are seen as too.
"""

# feature: (weight, default relative tolerance)
WEIGHTS = {"modulation": (2.0, None), "baud": (3.0, 0.005), "chip_rate": (3.0, 0.005), "code_length": (2.0, 0.0),
           "bt": (1.0, 0.15), "h": (1.0, 0.1), "beta": (0.5, 0.2), "scs": (3.0, 0.01), "cp": (2.0, 0.03),
           "pilot_spacing": (1.5, 0.0), "pilot_step": (1.0, 0.0), "pilot_period": (1.5, 0.0), "frame": (2.0, 0.01),
           "slots": (1.0, 0.0), "bandwidth": (0.5, 0.25), "frequency": (0.5, 0.0)}

def _sys(name, ref, **f):
    return {"name": name, "ref": ref, **f}

M = 1e6
k = 1e3
SYSTEMS = [
    # Cellular and trunked radio
    _sys("GSM", "3GPP TS 45.004, 45.002", modulation={"gmsk", "msk", "2fsk"}, baud=13e6/48, bt=0.3, h=0.5,
         frame=60/13*1e-3, slots=8, bandwidth=200*k, frequency=[(824*M, 960*M), (1710*M, 1990*M)]),
    _sys("EDGE (8PSK slots of GSM)", "3GPP TS 45.004", modulation={"8psk"}, baud=13e6/48, frame=60/13*1e-3, slots=8,
         bandwidth=200*k, frequency=[(824*M, 960*M), (1710*M, 1990*M)]),
    _sys("TETRA", "ETSI EN 300 392-2", modulation={"pi4dqpsk", "8psk", "dqpsk"}, baud=18*k, beta=0.35,
         frame=170/3*1e-3, slots=4, bandwidth=25*k, frequency=[(380*M, 470*M)]),
    _sys("DMR", "ETSI TS 102 361-1", modulation={"4fsk", "c4fm"}, baud=4800, frame=60e-3, slots=2, bandwidth=12.5*k),
    _sys("dPMR / NXDN 6.25 kHz", "ETSI TS 102 658; NXDN TS 1-A", modulation={"4fsk"}, baud=2400, bandwidth=6.25*k),
    _sys("NXDN 12.5 kHz", "NXDN TS 1-A", modulation={"4fsk"}, baud=4800, bandwidth=12.5*k),
    _sys("APCO P25 phase 1", "TIA-102.BAAA", modulation={"4fsk", "c4fm"}, baud=4800, bandwidth=12.5*k),
    _sys("APCO P25 phase 2", "TIA-102.BBAB", modulation={"hdqpsk", "pi4dqpsk", "8psk", "hcpm", "4fsk"}, baud=6000,
         slots=2, bandwidth=12.5*k),
    _sys("LTE", "3GPP TS 36.211", modulation={"ofdm"}, scs=15*k, cp=[144/2048/15e3, 160/2048/15e3, 512/2048/15e3],
         pilot_spacing=[6, 3], pilot_period=[140, 7], frame=[10e-3, 5e-3]),
    _sys("5G NR", "3GPP TS 38.211", modulation={"ofdm"}, scs=[15*k, 30*k, 60*k, 120*k],
         cp=[144/2048/15e3, 144/2048/30e3, 144/2048/60e3, 144/2048/120e3], frame=[10e-3, 5e-3, 2.5e-3]),
    # Wireless LAN / PAN
    _sys("Wi-Fi 802.11a/g/n/ac (OFDM)", "IEEE 802.11-2020 17", modulation={"ofdm"}, scs=312.5*k, cp=[0.8e-6, 0.4e-6],
         bandwidth=[16.6*M, 35.6*M, 75.8*M], frequency=[(2400*M, 2500*M), (5150*M, 5900*M)]),
    _sys("Wi-Fi 802.11ax", "IEEE 802.11ax-2021", modulation={"ofdm"}, scs=78.125*k, cp=[0.8e-6, 1.6e-6, 3.2e-6],
         frequency=[(2400*M, 2500*M), (5150*M, 7125*M)]),
    _sys("Wi-Fi 802.11b (DSSS)", "IEEE 802.11-2020 16", modulation={"bpsk", "qpsk", "dbpsk", "dqpsk"}, baud=1*M,
         chip_rate=11*M, code_length=11, bandwidth=22*M, frequency=[(2400*M, 2500*M)]),
    _sys("Bluetooth BR", "Bluetooth Core 5.4 vol 2 A", modulation={"gfsk", "2fsk"}, baud=1*M, bt=0.5, h=(0.28, 0.35),
         bandwidth=1*M, frequency=[(2400*M, 2483.5*M)]),
    _sys("Bluetooth LE 1M", "Bluetooth Core 5.4 vol 6 A", modulation={"gfsk", "gmsk", "2fsk"}, baud=1*M, bt=0.5,
         h=(0.45, 0.55), bandwidth=2*M, frequency=[(2400*M, 2483.5*M)]),
    _sys("Bluetooth LE 2M", "Bluetooth Core 5.4 vol 6 A", modulation={"gfsk", "gmsk", "2fsk"}, baud=2*M, bt=0.5,
         h=(0.45, 0.55), bandwidth=2*M, frequency=[(2400*M, 2483.5*M)]),
    _sys("IEEE 802.15.4 2.4 GHz (Zigbee, Thread)", "IEEE 802.15.4-2020 12", modulation={"oqpsk", "msk", "qpsk"},
         chip_rate=2*M, code_length=32, bandwidth=2*M, frequency=[(2400*M, 2483.5*M)]),
    _sys("DECT", "ETSI EN 300 175-2", modulation={"gfsk", "gmsk", "2fsk"}, baud=1.152*M, bt=0.5, h=0.5, frame=10e-3,
         slots=24, bandwidth=1.728*M, frequency=[(1880*M, 1930*M)]),
    _sys("LoRa", "Semtech AN1200.22", modulation={"css"}, bandwidth=[125*k, 250*k, 500*k],
         frequency=[(433*M, 435*M), (863*M, 870*M), (902*M, 928*M)]),
    # Aviation, maritime, paging
    _sys("Mode S / ADS-B", "ICAO Annex 10 vol IV", modulation={"ppm"}, baud=1*M, frequency=[(1089*M, 1091*M)]),
    _sys("AIS", "ITU-R M.1371", modulation={"gmsk", "msk", "2fsk"}, baud=9600, bt=0.4, h=0.5, frame=60.0,
         slots=2250, bandwidth=25*k, frequency=[(161.9*M, 162.1*M)]),
    _sys("ACARS", "ARINC 618", modulation={"msk", "2fsk", "am"}, baud=2400, frequency=[(129*M, 137*M)]),
    _sys("VDL mode 2", "ICAO Annex 10 vol III", modulation={"d8psk", "8psk"}, baud=10.5*k, bandwidth=25*k,
         frequency=[(118*M, 137*M)]),
    _sys("POCSAG", "ITU-R M.584", modulation={"2fsk"}, baud=[512, 1200, 2400], h=(3.7, 18.0)),
    _sys("FLEX", "Motorola FLEX", modulation={"2fsk", "4fsk"}, baud=[1600, 3200]),
    # Broadcast
    _sys("DVB-T", "ETSI EN 300 744", modulation={"ofdm"}, scs=[1/224e-6, 1/896e-6, 1/256e-6, 1/1024e-6,
         1/298.67e-6, 1/1194.67e-6], pilot_spacing=[12, 3], pilot_step=3, pilot_period=4,
         frequency=[(174*M, 230*M), (470*M, 862*M)]),
    _sys("ISDB-T", "ARIB STD-B31", modulation={"ofdm"}, scs=[1/252e-6, 1/504e-6, 1/1008e-6], pilot_spacing=[12, 3],
         pilot_step=3, pilot_period=4, frequency=[(470*M, 806*M)]),
    _sys("DAB", "ETSI EN 300 401", modulation={"ofdm", "dqpsk"}, scs=[1*k, 4*k, 8*k, 2*k],
         cp=[246e-6, 62e-6, 31e-6, 123e-6], frame=[96e-3, 24e-3, 48e-3], frequency=[(174*M, 240*M), (1452*M, 1492*M)]),
    _sys("ATSC 1.0 (8-VSB)", "ATSC A/53", modulation={"8vsb"}, baud=4.5e6/286*684, bandwidth=6*M,
         frequency=[(54*M, 698*M)]),
    _sys("FM broadcast", "ITU-R BS.450", modulation={"fm"}, bandwidth=(150*k, 260*k), frequency=[(76*M, 108*M)]),
    # Satellite
    _sys("DVB-S", "ETSI EN 300 421", modulation={"qpsk"}, beta=0.35, frequency=[(10.7e9, 12.75e9), (3.4e9, 4.2e9)]),
    _sys("DVB-S2", "ETSI EN 302 307-1", modulation={"qpsk", "8psk", "16apsk", "32apsk"}, beta=[0.35, 0.25, 0.2],
         frequency=[(10.7e9, 12.75e9), (3.4e9, 4.2e9)]),
    _sys("Iridium", "Iridium (ITU filings)", modulation={"dqpsk", "qpsk", "bpsk"}, baud=25*k, beta=0.4, frame=90e-3,
         bandwidth=41.667*k, frequency=[(1616*M, 1626.5*M)]),
    _sys("GPS L1 C/A", "IS-GPS-200", modulation={"bpsk"}, chip_rate=1.023*M, code_length=1023, baud=50,
         frequency=[(1575.0*M, 1576.0*M)]),
    _sys("Galileo E1 OS", "Galileo OS SIS ICD", modulation={"boc", "bpsk"}, chip_rate=1.023*M, code_length=4092,
         frequency=[(1575.0*M, 1576.0*M)]),
    _sys("GLONASS L1 OF", "GLONASS ICD", modulation={"bpsk"}, chip_rate=0.511*M, code_length=511,
         frequency=[(1598*M, 1606*M)]),
    _sys("CCSDS TT&C (PM, subcarrier)", "CCSDS 401.0-B", modulation={"pm"}, frequency=[(2025*M, 2300*M),
         (7145*M, 8500*M)]),
    # Telemetry
    _sys("IRIG 106 SOQPSK-TG", "IRIG 106 ch 2", modulation={"soqpsk", "oqpsk", "qpsk"}, h=0.5,
         frequency=[(1435*M, 1525*M), (2200*M, 2395*M), (4400*M, 4950*M)]),
    _sys("IRIG 106 PCM/FM", "IRIG 106 ch 2", modulation={"2fsk", "gfsk", "fm"}, h=0.7,
         frequency=[(1435*M, 1525*M), (2200*M, 2395*M)]),
]

ALIASES = {"gmsk_0.25": "gmsk", "gmsk_0.3": "gmsk", "gmsk_0.5": "gmsk", "gmsk_diff_0.3": "gmsk", "soqpsk_mil": "soqpsk",
           "soqpsk_tg": "soqpsk", "2-fsk": "2fsk", "4-fsk": "4fsk", "8-fsk": "8fsk", "2-gfsk": "gfsk", "4-gfsk": "4fsk",
           "pi/4-dqpsk": "pi4dqpsk", "16qam": "16qam", "o-qpsk": "oqpsk"}

def family(name):
    """A modulation name from any tool as the catalogue's family name (lower case, aliases folded)"""
    n = str(name).strip().lower()
    return ALIASES.get(n, n)

def _agrees(key, want, got, tol):
    """(agrees, how it reads)"""
    if key == "modulation":
        g = family(got)
        return g in want, f"{g}"
    if key == "frequency":
        ok = any(lo <= got <= hi for lo, hi in want)
        return ok, f"{got/1e6:.3f} MHz {'in' if ok else 'outside'} its bands"
    if isinstance(want, tuple):
        lo, hi = want
        ok = lo*(1 - tol) <= got <= hi*(1 + tol)
        return ok, f"{got:.4g} in {lo:.4g}..{hi:.4g}" if ok else f"{got:.4g} not in {lo:.4g}..{hi:.4g}"
    opts = want if isinstance(want, list) else [want]
    for v in opts:
        if (v == 0 and got == 0) or (v != 0 and abs(got - v) <= max(tol*abs(v), 1e-12 if tol else 0.5)):
            return True, f"{got:.6g} ~ {v:.6g}"
    return False, f"{got:.6g}, not {' / '.join(f'{v:.6g}' for v in opts)}"

class Match:
    """`match`: name, ref, score (-1..1), agree [(feature, reading)], contradict [(feature, reading)]"""
    __slots__ = ("name", "ref", "score", "agree", "contradict")
    def __init__(self, **kw):
        for k_ in self.__slots__:
            setattr(self, k_, kw.get(k_))

    def __str__(self):
        a = ", ".join(f"{f} {r}" for f, r in self.agree) or "nothing"
        c = "; against: " + ", ".join(f"{f} {r}" for f, r in self.contradict) if self.contradict else ""
        return f"{self.name} ({self.ref}): {self.score:.2f} - agrees: {a}{c}"

    __repr__ = __str__

def match(feats, systems=None, tolerance=None, top=None, prior=2.0):
    """Systems ranked by how well measured features fit (see the module): [Match], best first; systems no feature
    could be compared on are left out. tolerance: {feature: relative tolerance} over the defaults (a coarse measurement
    deserves a wider one)"""
    tol = {k_: t for k_, (_, t) in WEIGHTS.items()}
    tol.update(tolerance or {})
    feats = {k_: v for k_, v in feats.items() if v is not None}
    out = []
    for s in systems or SYSTEMS:
        got = most = 0.0
        agree, against = [], []
        for key, val in feats.items():
            if key not in s or key not in WEIGHTS:
                continue
            w = WEIGHTS[key][0]
            ok, how = _agrees(key, s[key], val, tol[key] or 0.0)
            most += w
            if ok:
                got += w
                agree.append((key, how))
            else:
                got -= w
                against.append((key, how))
        if most:
            out.append(Match(name=s["name"], ref=s["ref"], score=got/(most + prior), agree=agree, contradict=against))
    out.sort(key=lambda m: (-m.score, -len(m.agree)))
    return out[:top] if top else out

def features(det=None, fsk=None, cpm=None, ofdm=None, pilots=None, bursts=None, spread=None, frequency=None, **more):
    """The catalogue's features from rpps.analysis results: det (modulation.Detection), fsk (fsk.FSK), cpm (cpm.CPM or a
    scheme name), ofdm (waveform.OFDM), pilots (ofdm_pilots.Pilots), bursts (burst.Bursts or waveform.Bursts), spread
    ((chip rate, code length)), frequency (Hz); more: any feature by name (it wins)"""
    f = {}
    if det is not None:
        f.update(modulation=det.mod, baud=det.baud, beta=det.beta)
    if fsk is not None and fsk.M is not None:
        f.update(baud=fsk.baud, h=fsk.h, modulation=("gmsk" if fsk.M == 2 and abs(fsk.h - 0.5) < 0.06 and fsk.gaussian
                 else "msk" if fsk.M == 2 and abs(fsk.h - 0.5) < 0.06 else "gfsk" if fsk.M == 2 and fsk.gaussian
                 else f"{fsk.M}fsk"))
    if cpm is not None:
        name = cpm if isinstance(cpm, str) else cpm.scheme
        f["modulation"] = family(name)
        if not isinstance(cpm, str):
            f["baud"] = cpm.baud
        if "gmsk" in name:
            f["bt"] = float(name.rsplit("_", 1)[1])
        f["h"] = 0.5
    if ofdm is not None:
        f["modulation"] = "ofdm"
        if ofdm.comb:
            f["scs"] = 1/(ofdm.comb*ofdm.tu)
        else:
            f.update(scs=1/ofdm.tu, cp=ofdm.tcp)
    if pilots is not None and pilots.period:
        f["pilot_period"] = pilots.period
        if pilots.spacing:
            f["pilot_spacing"] = pilots.spacing
        if pilots.step:
            f["pilot_step"] = pilots.step
        g = pilots.grid
        f.setdefault("scs", g.fs/g.N)
        f.setdefault("cp", g.ncp/g.fs)
    if bursts is not None:
        fr = getattr(bursts, "frame", None) or getattr(bursts, "period", None)
        if fr:
            f["frame"] = fr
        sl = getattr(bursts, "slots", None)
        if sl:
            f["slots"] = len(sl)
        if getattr(bursts, "mod", None):
            f.setdefault("modulation", bursts.mod)
            f.setdefault("baud", bursts.baud)
    if spread is not None:
        f["chip_rate"], f["code_length"] = spread
    if frequency is not None:
        f["frequency"] = frequency
    f.update(more)
    return f

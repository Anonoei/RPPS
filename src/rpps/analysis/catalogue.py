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
  burst        True for bursts (TDMA, MF-TDMA, packets), False for a continuous carrier
  sync         the synchronisation signal seen: pss_sss (LTE), ssb (NR); ssb_period (s) an NR SSB burst period
  subcarrier, subcarrier_baud    a PM telemetry subcarrier (Hz) and its symbol rate (rp.analysis.framing.telemetry)
  profile      the shipped profile that decoded it (rp.analysis.profile's profile_name): strong evidence
  modulation   cw: an unmodulated carrier or a beacon (the waveform step's tones)

Each system lists values (a number, a list of allowed numbers, a (lo, hi) range, or a set of names) with a relative
tolerance. A feature both sides have scores +weight when it agrees and -weight when it contradicts; the score is the
sum over the most it could have been plus `prior` (so a fit on one feature ranks below a fit on several: 1 is
approached as more agrees; below 0 more contradicts than agrees), and the reasons say which. Nothing measured that a
system lacks counts either way. A positive score needs two agreeing features that are not weak (frequency,
bandwidth), or one strong one (baud, chip_rate, scs, code_length, profile, subcarrier): a carrier's frequency alone,
or a modulation alone, names nothing. Systems the features can't tell apart are reported `tied`, not ordered. A blind tool reads some modulations as
others (pi/4-DQPSK as 8PSK, GMSK as MSK or 2-FSK, C4FM as 4-FSK): systems list what they are seen as too.
"""

from .. import _config
from ..errors import ConfigError

# feature: (weight, default relative tolerance)
WEIGHTS = {"modulation": (2.0, None), "baud": (3.0, 0.005), "chip_rate": (3.0, 0.005), "code_length": (2.0, 0.0),
           "bt": (1.0, 0.15), "h": (1.0, 0.1), "beta": (0.5, 0.2), "scs": (3.0, 0.01), "cp": (2.0, 0.03),
           "pilot_spacing": (1.5, 0.0), "pilot_step": (1.0, 0.0), "pilot_period": (1.5, 0.0), "frame": (2.0, 0.01),
           "slots": (1.0, 0.0), "bandwidth": (0.5, 0.25), "frequency": (0.5, 0.0), "burst": (1.5, None),
           "sync": (2.0, None), "profile": (3.0, None), "ssb_period": (2.0, 0.01), "subcarrier": (2.0, 0.01),
           "subcarrier_baud": (2.0, 0.01)}
CATEGORICAL = ("modulation", "burst", "sync", "profile") # Compared by name (or True / False), not by number
STRONG = ("baud", "chip_rate", "scs", "code_length", "profile", "subcarrier") # One of these agreeing is evidence alone
WEAK = ("frequency", "bandwidth") # Never enough on their own

def _value(key, v):
    """A feature as config/system writes it, as `match` reads it: a list of names a set, {"range": [lo, hi]} a tuple,
    frequency bands tuples"""
    if key == "modulation":
        return {family(x) for x in (v if isinstance(v, list) else [v])}
    if key in ("sync", "profile"):
        return {str(x) for x in (v if isinstance(v, list) else [v])}
    if key == "frequency":
        return [tuple(b) for b in v]
    if isinstance(v, dict) and "range" in v:
        return tuple(v["range"])
    return v

def load_system(name, obj):
    """A config/system definition as `match` takes it: {"name": its title, "ref", "id": its reference, "profile",
    "tags", and each feature}"""
    unknown = set(obj.get("features") or {}) - set(WEIGHTS)
    if unknown:
        raise ConfigError(f"system/{name}: no features {sorted(unknown)} (has {sorted(WEIGHTS)})")
    return {"name": obj.get("title", name), "ref": obj.get("ref"), "id": name, "profile": obj.get("profile"),
            "tags": list(obj.get("tags") or []),
            **{k_: _value(k_, v) for k_, v in (obj.get("features") or {}).items()}}

DOMAIN = _config.register_domain("system", {"system": load_system})

_SYSTEMS = {}

def systems(fresh=False):
    """Every system in config/system (user layers too), loaded once: [dict], in reference order"""
    key = tuple(str(p) for p in _config.layers("system"))
    if fresh or key not in _SYSTEMS:
        _SYSTEMS.clear()
        _SYSTEMS[key] = [_config.load("system", f"{cat}/{n}") for cat, names in _config.ls("system").items() for n in names]
    return _SYSTEMS[key]

_configured = systems # `match` takes a `systems` argument

ALIASES = {"tone": "cw", "carrier": "cw", "unmodulated": "cw", "gmsk_0.25": "gmsk", "gmsk_0.3": "gmsk", "gmsk_0.5": "gmsk", "gmsk_diff_0.3": "gmsk", "soqpsk_mil": "soqpsk",
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
    if key == "burst":
        return bool(got) == bool(want), "bursts" if got else "continuous"
    if key == "sync":
        return str(got) in want, f"{got}"
    if key == "profile": # Its family decoded it: any variant of the profile (dvbs/qpsk_3_4 for dvbs/qpsk)
        fam = str(got).split("/")[0]
        wants = {want} if isinstance(want, str) else want
        return any(fam == w.split("/")[0] for w in wants), f"{got} decodes"
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
    """`match`: name, ref, score (-1..1), agree [(feature, reading)], contradict [(feature, reading)], tied (the names
    of other systems the features can't tell this one from: the same score and agreement)"""
    __slots__ = ("name", "ref", "score", "agree", "contradict", "tied")
    def __init__(self, **kw):
        for k_ in self.__slots__:
            setattr(self, k_, kw.get(k_))

    def __str__(self):
        a = ", ".join(f"{f} {r}" for f, r in self.agree) or "nothing"
        c = "; against: " + ", ".join(f"{f} {r}" for f, r in self.contradict) if self.contradict else ""
        t = f" [ambiguous: tied with {', '.join(self.tied)}]" if self.tied else ""
        return f"{self.name} ({self.ref}): {self.score:.2f} - agrees: {a}{c}{t}"

    __repr__ = __str__

def match(feats, systems=None, tolerance=None, top=None, prior=2.0):
    """Systems ranked by how well measured features fit (see the module): [Match], best first; systems no feature
    could be compared on are left out. tolerance: {feature: relative tolerance} over the defaults (a coarse measurement
    deserves a wider one)"""
    tol = {k_: t for k_, (_, t) in WEIGHTS.items()}
    tol.update(tolerance or {})
    feats = {k_: v for k_, v in feats.items() if v is not None}
    out = []
    for s in systems if systems is not None else _configured():
        got = most = 0.0
        agree, against = [], []
        for key, val in feats.items():
            if s.get(key) is None or key not in WEIGHTS:
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
            score = got/(most + prior)
            # Weak evidence is no match: two features that aren't weak must agree, or one strong one
            firm = [f for f, _ in agree if f not in WEAK]
            alone = family(feats.get("modulation", "")) == "cw" # A carrier alone is what it is
            if score > 0 and len(firm) < 2 and not any(f in STRONG for f in firm) and not alone:
                score = 0.0
            out.append(Match(name=s["name"], ref=s["ref"], score=score, agree=agree, contradict=against, tied=[]))
    out.sort(key=lambda m: (-m.score, -len(m.agree), m.name))
    for m in out: # The same score on the same features: say so rather than let the name order decide
        if m.score > 0:
            m.tied = [o.name for o in out if o is not m and abs(o.score - m.score) < 1e-9 and
                      sorted(f for f, _ in o.agree) == sorted(f for f, _ in m.agree)]
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
    if bursts is not None and not isinstance(bursts, bool):
        f["burst"] = True
        fr = getattr(bursts, "frame", None) or getattr(bursts, "period", None)
        if fr:
            f["frame"] = fr
        sl = getattr(bursts, "slots", None)
        if sl:
            f["slots"] = len(sl)
        if getattr(bursts, "mod", None):
            f.setdefault("modulation", bursts.mod)
            f.setdefault("baud", bursts.baud)
    if isinstance(bursts, bool): # Measured as bursty, or as continuous
        f["burst"] = bursts
    if spread is not None:
        f["chip_rate"], f["code_length"] = spread
    if frequency is not None:
        f["frequency"] = frequency
    f.update(more)
    return f

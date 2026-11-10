"""RDS link layer: differential bits in, groups (and PI / PS / RadioText / clock time) out

RPPS recovers the biphase symbols; this turns them into blocks. A block is 16 data bits plus 10 check bits whose
syndrome, once the block's offset word (A, B, C, C', D) is mixed in, is a known value. So a block is "valid" when
its syndrome equals one of five constants, and four valid blocks A, B, C|C', D in a row are one group.
"""
from collections import Counter

POLY = 0x5B9 # x^10 + x^8 + x^7 + x^5 + x^4 + x^3 + 1
OFFSETS = {"A": 0x0FC, "B": 0x198, "C": 0x168, "C'": 0x350, "D": 0x1B4}
NEXT = {"A": ("B",), "B": ("C", "C'"), "C": ("D",), "C'": ("D",), "D": ("A",)}

PTY = ("None", "News", "Information", "Sports", "Talk", "Rock", "Classic Rock", "Adult Hits", "Soft Rock", "Top 40",
       "Country", "Oldies", "Soft", "Nostalgia", "Jazz", "Classical", "R&B", "Soft R&B", "Language", "Religious Music",
       "Religious Talk", "Personality", "Public", "College", "Spanish Talk", "Spanish Music", "Hip Hop", "Unassigned",
       "Unassigned", "Weather", "Emergency Test", "Emergency")

def syndrome(word):
    """The 10-bit syndrome of a 26-bit block"""
    r = 0
    for i in range(25, -1, -1):
        r = (r << 1) | ((word >> i) & 1)
        if r & 0x400:
            r ^= POLY
    return r

SYNDROMES = {syndrome(v): k for k, v in OFFSETS.items()}

def _text(chars):
    return "".join(c if 32 <= ord(c) < 127 else "?" for c in chars)

class RDS:
    """Feed it differentially-decoded bits; read the station's data

    bits(syms) takes the slicer output (+ / - symbols, the Costas loop's sign ambiguity doesn't matter) and does the
    differential decoding. events() returns what changed since it was last called, as printable strings.
    """
    def __init__(self, lose=12):
        self.reg = 0 # The last 26 bits
        self.n = 0 # Bits seen
        self.prev = 0
        self.synced = False
        self.expect = None # Offset name of the next block
        self.count = 0 # Bits since the last block boundary
        self.bad = 0
        self.lose = lose
        self.blocks = []
        self.groups = 0
        self.errors = 0
        self.pi = None
        self.ps = [" "]*8
        self.rt = [None]*64 # None: not received (yet)
        self.rt_ab = None
        self.ps_names = Counter() # Names that held steady for two full cycles: stations rotate through several
        self.ps_last = None
        self.ps_distinct = set() # Every whole name seen: a static name is one, a scrolling one is many
        self._ps_prev = None
        self.pty = None
        self.tp = None
        self.ct = None
        self._seen = {}
        self._ps_mask = 0
        self._events = []
        self._pis = Counter()

    def bits(self, syms):
        for s in syms:
            b = 1 if s > 0 else 0
            self.push(b ^ self.prev)
            self.prev = b

    def push(self, bit):
        self.reg = ((self.reg << 1) | bit) & 0x3FFFFFF
        self.n += 1
        if self.synced:
            self.count += 1
            if self.count < 26:
                return
            self.count = 0
            name = SYNDROMES.get(syndrome(self.reg))
            if name in NEXT.get(self.expect, ()) or name == self.expect:
                self.bad = max(0, self.bad - 1)
                self._block(name, self.reg >> 10)
                self.expect = name
            else:
                self.bad += 1
                self.errors += 1
                self.blocks = []
                self.expect = {"A": "B", "B": "C", "C": "D", "C'": "D", "D": "A"}[self.expect]
                if self.bad >= self.lose:
                    self.synced = False
                    self.blocks = []
        else:
            if self.n < 26:
                return
            name = SYNDROMES.get(syndrome(self.reg))
            if name is None:
                return
            # Two valid blocks 26 bits apart, in order, make a sync
            if getattr(self, "_cand", None) and self._cand[1] == self.n - 26 and name in NEXT[self._cand[0]]:
                self.synced, self.count, self.bad, self.expect = True, 0, 0, name
                self.blocks = []
                self._block(name, self.reg >> 10)
            self._cand = (name, self.n)

    def _block(self, name, data):
        if name == "A":
            self.blocks = [data]
        elif self.blocks and len(self.blocks) == {"B": 1, "C": 2, "C'": 2, "D": 3}[name]:
            self.blocks.append(data)
            if name == "D":
                self._group(self.blocks)
                self.blocks = []
        else:
            self.blocks = []

    def _set(self, key, value, label):
        if self._seen.get(key) != value:
            self._seen[key] = value
            self._events.append(f"{label}: {value}")

    def _group(self, g):
        a, b, c, d = g
        self.groups += 1
        self._pis[a] += 1
        self.pi = self._pis.most_common(1)[0][0]
        self.tp = (b >> 10) & 1
        self.pty = (b >> 5) & 31
        gtype, ver = b >> 12, (b >> 11) & 1
        if gtype == 0: # Programme service name, two characters at a time
            i = b & 3
            self.ps[2*i], self.ps[2*i + 1] = chr(d >> 8), chr(d & 255)
            self._ps_mask |= 1 << i
            if self._ps_mask == 15: # All four segments are fresh: a whole name, not half of two
                self._set("ps", _text(self.ps), "PS")
                name = _text(self.ps).strip()
                self.ps_last = name
                self.ps_distinct.add(name)
                if name == self._ps_prev: # Seen whole twice running: not a name half-changed into another
                    self.ps_names[name] += 1
                self._ps_prev = name
                self._ps_mask = 0
        elif gtype == 2: # RadioText
            ab = (b >> 4) & 1
            if self.rt_ab is not None and ab != self.rt_ab:
                self.rt = [None]*64 # The flag toggles when the text is replaced
            self.rt_ab = ab
            i = b & 15
            chars = (c >> 8, c & 255, d >> 8, d & 255) if ver == 0 else (d >> 8, d & 255)
            step = 4 if ver == 0 else 2
            for k, ch in enumerate(chars):
                if step*i + k < 64:
                    self.rt[step*i + k] = chr(ch)
            self._set("rt", self.rt_text, "RT")
        elif gtype == 4 and ver == 0: # Clock time
            mjd = ((b & 3) << 15) | (c >> 1)
            hour = ((c & 1) << 4) | (d >> 12)
            minute = (d >> 6) & 63
            off = (d & 31) * (-1 if d & 32 else 1) * 0.5
            self._set("ct", f"MJD {mjd} {hour:02d}:{minute:02d} UTC{off:+g}", "Clock")
        self._set("pty", PTY[self.pty], "Programme type")
        self._set("pi", f"{self.pi:04X}", "PI")

    @property
    def ps_text(self):
        """The station name: the steadiest one seen most often, else the last whole one, else what has arrived so far"""
        if self.ps_names:
            return self.ps_names.most_common(1)[0][0]
        return self.ps_last or _text(self.ps).strip()

    @property
    def ps_scrolls(self):
        """The station changes its name field over time (scrolling song titles, rotating slogans): no single frame is its name"""
        return len(self.ps_distinct) >= 3

    @property
    def rt_text(self):
        """RadioText up to its end marker (0x0D); characters not received yet show as _"""
        chars = []
        for c in self.rt:
            if c == "\r":
                break
            chars.append("_" if c is None else c)
        return _text(chars).rstrip(" _")

    @property
    def pi_count(self):
        """How many groups carried the PI we settled on: a handful of matching ones is a station, one is noise"""
        return self._pis[self.pi] if self.pi is not None else 0

    def events(self):
        out, self._events = self._events, []
        return out

    def reliable(self, groups=5):
        """True once enough groups agreed on one PI that this is a real RDS signal, not a chance block match"""
        return self.pi_count >= groups and self.pi not in (0, 0xFFFF) # 0000 is what a noisy block match tends to give

    def __str__(self):
        sync = "sync" if self.synced else "searching"
        pi = f"{self.pi:04X}" if self.pi is not None else "----"
        return f"RDS {sync}, PI {pi}, PS '{_text(self.ps)}', {self.groups} groups"

"""FEC chains: several configured stages run in order, defined in config/chain/<name>.json

A chain lists stages by kind and config name:
    {"conv": "dvbs.3_4_7"}                          config/coding/conv      (decode: Viterbi)
    {"align": {"markers": [71, 184], "period": 204}} finds byte alignment from repeating sync bytes
    {"interleave": "forney.12_17"}                  config/coding/interleave
    {"block": "rs.204_188"}                         config/coding/block    (a Reed-Solomon code here)
    {"scram": ["adt", "dvb_energy"]}                config/scram/<folder>/<name>
    {"crc": "crc32_mpeg2"}                          config/encap/crc
The decode result is a (packets, bytes) array. Hard bits can be inverted by a symbol ambiguity: with
"polarity": "auto" both are tried and the one the block code accepts is kept.
"""
import json

import numpy as np

import json as _json
from pyboiler.config import config

from . import coding, encap
from .analysis import fec

class Result:
    __slots__ = ("packets", "stats")
    def __init__(self, packets, stats):
        self.packets, self.stats = packets, stats
    def __repr__(self):
        return f"<Result {len(self.packets)} packets {self.stats}>"

class Chain:
    def __init__(self, name, stages, polarity="auto", packet=None):
        self.name, self.stages, self.polarity, self.packet = name, stages, polarity, packet

    @staticmethod
    def load(name, path=None):
        from pathlib import Path
        base = Path(path) if path else config().PATH_CONFIG / "chain" # type: ignore
        obj = json.loads((base / f"{name}.json").read_text())
        return Chain(name, obj["stages"], obj.get("polarity", "auto"), obj.get("packet"))

    def _load(self, kind, ref):
        if kind == "scram":
            from . import scram
            return scram.load(*ref)
        if kind == "crc":
            return encap.load("crc", ref)
        return coding.load(kind, ref) # block | conv | interleave

    # --- decode ---
    def decode(self, bits, soft=None):
        """Run the chain on the received bits (or LLRs for the first conv stage). Returns a Result"""
        stats = {}
        stages = list(self.stages)
        x = np.asarray(bits)
        if "conv" in stages[0]:
            code = self._load("conv", stages[0]["conv"])
            enc = code._enc
            nb = int(enc.puncture.sum()) if enc.puncture is not None else enc.den
            off = 0
            if nb > 1 and len(x) > 20*nb*100: # Puncture phase: the alignment the decoder reproduces best
                L = enc.puncture.shape[1] if enc.puncture is not None else 1
                seg = len(x)//10//nb*nb # Multiple of the puncture period, so offsets are absolute
                err = []
                for o in range(nb):
                    s_ = np.asarray(x[seg+o:seg+o+(2500//L)*nb])
                    s_ = (s_ < 0).astype(np.uint8) if s_.dtype.kind == "f" else s_.astype(np.uint8)
                    k, n = fec._mismatch(code, s_)
                    err.append(k/n)
                off = int(np.argmin(err))
                stats["conv_offset"] = off
            x = x[off:]
            x = code._dec.decode(x, soft=soft) if x.dtype.kind == "f" or soft else code._dec.decode(x.astype(np.uint8))
            stats["viterbi_bits"] = len(x)
            stages = stages[1:]
        best = None
        for pol in ((0, 1) if self.polarity == "auto" else (int(self.polarity),)):
            res = self._bytes(np.asarray(x, dtype=np.uint8) ^ pol, stages)
            res.stats["polarity"] = pol
            if best is None or res.stats.get("rs_ok", 0) > best.stats.get("rs_ok", 0):
                best = res
        best.stats.update(stats)
        return best

    def _bytes(self, bits, stages):
        stats = {}
        data = bits
        arr = None
        for st in stages:
            kind, ref = next(iter(st.items()))
            if kind == "align":
                _, off, ph = fec.find_sync(data, ref["markers"], ref["period"])[0:3]
                frac = fec.find_sync(data, ref["markers"], ref["period"])[0]
                stats["sync"] = round(frac, 3)
                b = np.packbits(data[off:][:(len(data)-off)//8*8])
                arr = b[ph:].astype(np.int64)
            elif kind == "interleave":
                il = self._load("interleave", ref)._enc
                arr = il.decode(arr, symbols=True)
                arr = arr[(il.I - 1)*il.M*il.I:]
                arr = arr[:len(arr)//self.packet*self.packet] if self.packet else arr
            elif kind == "block":
                rs = self._load("block", ref)._enc
                n = rs.den
                arr = arr[:len(arr)//n*n]
                out = rs.decode(arr, symbols=True)
                stats["rs_ok"] = round(float(np.mean(rs.errors >= 0)), 4) if len(rs.errors) else 0.0
                stats["rs_corrected"] = int(rs.errors[rs.errors > 0].sum())
                stats["rs_failed"] = int((rs.errors < 0).sum())
                arr = out.reshape(-1, rs.num)
            elif kind == "scram":
                arr = self._descramble(arr, self._load("scram", ref), stats)
            elif kind == "crc":
                crc = self._load("crc", ref)._enc
                ok = [crc.to_bytes(crc.compute(p[:-crc.width//8])) == bytes(p[-crc.width//8:]) for p in arr]
                stats["crc_ok"] = round(float(np.mean(ok)), 4)
        return Result(arr, stats)

    @staticmethod
    def _descramble(arr, scram, stats):
        """Align on the scrambler's restart marker (first byte of a packet), descramble whole frames"""
        fb, pk = scram.frame["bytes"], scram.frame["sync_period"]
        per = fb//pk
        marker = scram.marker
        starts = np.nonzero(arr[:, 0] == marker)[0]
        if len(starts) == 0:
            stats["scram"] = "no restart marker found"
            return arr
        s0 = starts[0]
        groups = (len(arr) - s0)//per
        sel = arr[s0:s0 + groups*per].astype(np.uint8)
        bits = np.unpackbits(sel.reshape(-1))
        out = np.packbits(scram.decode(bits)).reshape(-1, sel.shape[1])
        out[::per, 0] = 0x47 # The inverted sync becomes a normal one again
        stats["frames"] = groups
        return out

    # --- encode ---
    def encode(self, packets):
        """Packets (N, bytes) -> coded bits (the chain run backwards)"""
        x = np.asarray(packets, dtype=np.uint8)
        for st in reversed(self.stages):
            kind, ref = next(iter(st.items()))
            if kind == "scram":
                sc = self._load("scram", ref)
                per = sc.frame["bytes"]//sc.frame["sync_period"]
                x = x[:len(x)//per*per].copy()
                x[::per, 0] = sc.marker
                x = np.packbits(sc.encode(np.unpackbits(x.reshape(-1)))).reshape(-1, x.shape[1])
            elif kind == "block":
                rs = self._load("block", ref)._enc
                x = rs.encode(x.reshape(-1), symbols=True).reshape(-1, rs.den).astype(np.uint8)
            elif kind == "interleave":
                il = self._load("interleave", ref)._enc
                x = il.encode(x.reshape(-1), symbols=True).astype(np.uint8)
            elif kind == "conv":
                code = self._load("conv", ref)
                bits = np.unpackbits(x.reshape(-1)) if x.dtype == np.uint8 else x
                return code._enc.encode(bits.astype(bool))
        return np.unpackbits(x.reshape(-1)).astype(bool)

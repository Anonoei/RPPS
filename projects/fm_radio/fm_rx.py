"""Live FM broadcast receiver: HackRF -> RPPS -> speakers, with RDS decoding

    python projects/fm_radio/fm_rx.py 105.1                    # lock to the station and play it (stereo when the pilot is there)
    python projects/fm_radio/fm_rx.py 105.1 --rds              # also decode RDS (station name, text, clock)
    python projects/fm_radio/fm_rx.py 105.1 --no-audio --rds   # decode only
    python projects/fm_radio/fm_rx.py 105.1 --wav out.wav      # record what is played
    python projects/fm_radio/fm_rx.py --scan                   # list the stations the HackRF can hear, then exit
    python projects/fm_radio/fm_rx.py --scan --rds             # ... and read each one's RDS (name, type, text)

The path:  HackRF 2.4 MS/s -> survey (find the carrier) -> shift + decimate to 480 kS/s -> FM demod = MPX
    MPX: mono (L+R, 0-15k) | 19k pilot -> PLL -> 38k carrier -> stereo (L-R) | 57k -> RDS (BPSK, 1187.5 baud)
The carrier is tracked (AFC) from the demodulator's DC, the stereo carrier from the pilot (PLL).
"""
import argparse
import queue
import shutil
import subprocess
import sys
import threading
import time
import wave
from types import SimpleNamespace

import numpy as np

import rpps as rp
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import _paths  # noqa: E402,F401
import radio_io
from radio_io import device
from rpps.signal.sample.down import interpolate_stream
from rds_link import RDS, PTY

FS = 2.4e6 # Capture rate: /5 -> 480 kS/s MPX, /10 -> 48 kS/s audio
AUDIO = 48000
BLOCK = 2**17 # Samples per block (55 ms)
OFFSET = 300e3 # Tune this far below the station: the HackRF's DC spike stays out of the signal
CHANNEL = 200e3 # An FM broadcast channel
PILOT = 19e3
RDS_BAUD = 1187.5

# ---------------------------------------------------------------------------------------------- stages

class PilotPLL(rp.base.Stage):
    """Mix the MPX down to the stereo difference signal, locking the 38 kHz carrier to the 19 kHz pilot

    A block-rate PLL: each block measures the pilot's phase against the oscillator, then corrects the oscillator's
    phase and frequency. The pilot is a steady tone, so 50 corrections a second hold it. Output is (L-R) before
    low-pass filtering; `st.level` is the pilot amplitude (MPX units) and `st.err` the last phase error [rad].
    """
    def __init__(self, kp=0.3, ki=0.04):
        self.kp, self.ki = kp, ki

    def new_state(self, meta):
        return SimpleNamespace(phase=0.0, freq=2*np.pi*PILOT/meta.Fs, level=0.0, err=0.0, locked=False)

    def step(self, x, st, meta):
        n = len(x)
        if n == 0:
            return x
        k = np.arange(n)
        ph = st.phase + st.freq*k
        z = np.sum(x * np.exp(-1j*ph)) / n
        st.level = 2*abs(z)
        st.err = float(np.angle(z))
        # Ignore the correction when there is no pilot: a station without one leaves noise here
        if st.level > 0.005:
            st.freq += self.ki*st.err/n
            st.phase += self.kp*st.err
        st.locked = st.level > 0.015 and abs(st.err) < 0.5
        out = 2*x*np.cos(2*ph)
        st.phase = (st.phase + st.freq*n) % (2*np.pi)
        return out.astype(np.float32)

class Deemphasis(rp.signal.filters.IIR):
    """Broadcast FM boosts the highs: one pole at 1/(2*pi*tau) takes the boost back out (75 us US, 50 us elsewhere)"""
    def __init__(self, tau, fs):
        alpha = 1 - np.exp(-1/(tau*fs))
        super().__init__(rp.signal.filters.iir.single_pole(alpha))

# ---------------------------------------------------------------------------------------------- output

class Speaker:
    """Raw 16-bit stereo to PipeWire (pw-play) or PulseAudio (paplay), and/or a WAV file

    The HackRF's sample clock and the sound card's are different crystals, and blocks arrive in 55 ms bursts, so playing
    "as it comes" runs dry (a dropout) or backs up. Instead: a cushion (`target` seconds) is written ahead, a writer thread
    takes the pipe's blocking off the receive loop, and the audio is resampled by a hair (under 1%) to hold the cushion
    steady. The cushion is measured as audio written minus wall time played.
    """
    def __init__(self, play=True, wav=None, target=0.25):
        self.proc = None
        self.target = target
        self.t0 = None # When playing began
        self.written = 0.0 # Seconds handed to the player, including silence
        self.underruns = 0
        self.produced = 0.0 # Seconds of real audio taken in (silence not counted)
        self.t_first = None
        self.step = 1.0 # Input samples per output sample
        self._st = [SimpleNamespace(tail=None, pos=1.0), SimpleNamespace(tail=None, pos=1.0)]
        self._q = queue.Queue()
        if play:
            if shutil.which("pw-play"):
                cmd = ["pw-play", "--raw", "--rate", str(AUDIO), "--channels", "2", "--format", "s16", "--latency", "100ms", "-"]
            elif shutil.which("paplay"):
                cmd = ["paplay", "--raw", f"--rate={AUDIO}", "--channels=2", "--format=s16le", "--latency-msec=100"]
            else:
                raise SystemExit("No audio player found (pw-play or paplay); use --wav or --no-audio")
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            self._thread = threading.Thread(target=self._drain, daemon=True)
            self._thread.start()
        self.wav = None
        if wav:
            self.wav = wave.open(wav, "wb")
            self.wav.setnchannels(2); self.wav.setsampwidth(2); self.wav.setframerate(AUDIO)

    def _drain(self):
        while True:
            pcm = self._q.get()
            if pcm is None:
                return
            try:
                self.proc.stdin.write(pcm)
                self.proc.stdin.flush()
            except (BrokenPipeError, ValueError):
                return

    @property
    def fill(self):
        """Seconds of audio the player has been given and not yet played"""
        return 0.0 if self.t0 is None else self.written - (time.monotonic() - self.t0)

    def _put(self, pcm, seconds):
        self._q.put(pcm)
        self.written += seconds

    def write(self, left, right):
        if self.t_first is None:
            self.t_first = time.monotonic()
        self.produced += len(left)/AUDIO
        if self.wav:
            both = (np.clip(np.stack((left, right), axis=1), -1, 1)*32767).astype("<i2")
            self.wav.writeframes(both.tobytes())
        if not self.proc:
            return
        if self.proc.poll() is not None:
            raise SystemExit("The audio player quit")
        if self.t0 is None: # Start with the cushion
            self.t0 = time.monotonic()
            self._put(bytes(int(self.target*AUDIO)*4), self.target)
        elif self.fill < 0: # Ran dry: that was a dropout. Rebuild the cushion
            self.underruns += 1
            self._put(bytes(int(self.target*AUDIO)*4), self.target)
        # A full cushion means the player is slower than the radio: take in a little more per output sample, and the reverse
        self.step = 1.0 + float(np.clip(0.02*(self.fill - self.target), -0.01, 0.01))
        l = interpolate_stream(np.asarray(left, dtype=np.float32), self._st[0], self.step, "cubic")
        r = interpolate_stream(np.asarray(right, dtype=np.float32), self._st[1], self.step, "cubic")
        pcm = (np.clip(np.stack((l, r), axis=1), -1, 1)*32767).astype("<i2")
        self._put(pcm.tobytes(), len(pcm)/AUDIO)

    def close(self):
        if self.wav:
            self.wav.close()
        if self.proc:
            self._q.put(None)
            self._thread.join(timeout=5)
            try:
                self.proc.stdin.close()
            except (BrokenPipeError, ValueError):
                pass
            try:
                self.proc.wait(timeout=5 + self.target)
            except subprocess.TimeoutExpired:
                self.proc.kill()

# ---------------------------------------------------------------------------------------------- acquire

def survey(sdr_args, cf, seconds=0.3):
    """Capture a moment on the air and find the signals in it. Returns the Survey"""
    src = rp.source.Device(device(), rx_gain=sdr_args.gain, cf=cf, Fs=FS)
    n, parts = 0, []
    for blk in src(BLOCK):
        parts.append(np.asarray(blk.data)); n += BLOCK
        if n >= max(2**18, seconds*FS):
            break
    return rp.analysis.survey(np.concatenate(parts), Fs=FS, cf=src.cf, more=0, baud=False)

def scan(args):
    """Step across the FM band in 2 MHz hops (one capture is 2.4 MHz wide) and list what is on the air

    With --rds, then tune to each station in turn and read its RDS: name, programme type, text, clock.
    """
    print("Scanning 88-108 MHz ...")
    found = []
    for c in np.arange(89e6, 108e6, 2e6):
        sv = survey(args, c, seconds=0.1)
        found += [s for s in sv if abs(s.freq - c) < 1e6 and 60e3 <= s.width <= 260e3 and (s.snr or 0) > 6
                  and abs(s.freq - c) > 20e3] # The DC spike sits at the tuned frequency
    found.sort(key=lambda s: s.freq)
    keep = [] # One station is one carrier: drop a weaker detection within 100 kHz of a stronger one
    for s in found:
        if keep and s.freq - keep[-1].freq < 100e3:
            if s.snr > keep[-1].snr:
                keep[-1] = s
        else:
            keep.append(s)
    if not args.rds:
        print("   freq MHz   width kHz   SNR dB")
        for s in keep:
            print(f"  {s.freq/1e6:9.3f}  {s.width/1e3:9.0f}  {s.snr:7.1f}")
        return
    print(f"{len(keep)} stations; reading RDS from each (up to {args.rds_time:g} s)\n")
    print("   freq MHz  SNR dB  audio   PI    name      type")
    done = []
    for s in keep:
        try:
            r = probe(args, s.freq, args.rds_time)
        except KeyboardInterrupt:
            print("(interrupted)")
            return
        pi = f"{r.pi:04X}" if r.pi is not None else "----"
        twin = next((f for f, p in done if p == r.pi and r.pi is not None and abs(f - s.freq) < 300e3), None)
        if twin: # The same programme within a channel or so: a leak from the stronger carrier, not a station
            r.note = f"same as {twin/1e6:.3f}"
        done.append((s.freq, r.pi))
        print(f"  {s.freq/1e6:9.3f}  {s.snr:6.1f}  {r.audio:6s}  {pi}  {r.ps or '':8s}  {r.pty or '':14s} {r.note}")
        if r.rt:
            print(f"{'':>45}text: {r.rt}")
        if r.clock:
            print(f"{'':>45}clock: {r.clock}")

def probe(args, carrier, seconds):
    """Tune to a carrier and read its RDS. Returns a namespace: audio (stereo/mono), pi, ps, pty, rt, clock, note

    Stops once it has the name and text, when `seconds` is up, or early (4 s) when there is no RDS to find.
    """
    st = build(args, carrier)
    src = rp.source.Device(device(), rx_gain=args.gain, cf=carrier - OFFSET, Fs=FS, sps=FS/RDS_BAUD)
    fo, afc, pilot = -OFFSET, 0.0, 0.0
    t0 = time.monotonic()
    dec = st.rds.decoder
    blocks = src(BLOCK)
    note = "timed out"
    try:
        for blk in blocks:
            mpx = blk + st.front + st.demod
            hz = float(np.mean(np.asarray(mpx.data)))/(0.5*2*np.pi)*mpx.Fs # AFC, as in run()
            afc += 0.1*(hz - afc)
            if abs(afc) > 50:
                fo -= 0.5*afc
                st.shift.fo = fo
                afc *= 0.5
            _ = mpx + st.pll
            pll = mpx.state(st.pll)
            if pll:
                pilot = max(pilot*0.8, pll.level) if pll.locked else pll.level*0.0 + pilot*0.8
            rb = mpx + st.rds.pre
            out = rb + st.rds.post
            if len(out):
                dec.bits(np.asarray(out.data).real)
            t = time.monotonic() - t0
            rt = dec.rt_text
            if t > 3.0 and dec.ps_names and rt and "_" not in rt: # Name steady, text with no gaps
                note = ""
                break
            if t > 4.0 and not dec.reliable():
                note = "no RDS" # A chance match of a few blocks is not a signal
                break
            if t > seconds:
                note = "partial" if dec.reliable() else "no RDS"
                break
    finally:
        blocks._gen.close() # Free the HackRF before the next station
    audio = "stereo" if pll and pll.level > 0.015 else "mono"
    ok = dec.reliable()
    if ok and dec.ps_scrolls:
        note = (note + " " if note else "") + "PS rotates"
    return SimpleNamespace(audio=audio, pi=dec.pi if ok else None, ps=dec.ps_text if ok and dec.ps_text else None,
                           pty=PTY[dec.pty] if ok and dec.pty else None, rt=dec.rt_text if ok and dec.rt_text else None,
                           clock=dec._seen.get("ct") if ok else None, note=note)

def acquire(args):
    """Find the station near args.freq. Returns its carrier [Hz]: the survey peak when there is one, else the dial frequency"""
    freq = args.freq*1e6
    sv = survey(args, freq - OFFSET)
    near = [s for s in sv if abs(s.freq - freq) < 120e3 and 60e3 < s.width < 300e3 and (s.snr or 0) > 6]
    if not near:
        print(f"No station found near {args.freq} MHz; using the dial frequency")
        return freq
    sig = max(near, key=lambda s: s.snr)
    # The survey's centre is the middle of its occupied band; FM is symmetric so that is the carrier
    print(f"Locked to {sig.freq/1e6:.4f} MHz ({sig.width/1e3:.0f} kHz wide, SNR {sig.snr:.1f} dB)")
    return sig.freq

# ---------------------------------------------------------------------------------------------- receive

def build(args, carrier):
    """The stages. Returns a namespace with the pipelines and the stages whose state we read"""
    front, Fs_mpx, D = rp.signal.channelize(0.0, CHANNEL, FS, oversample=2.39)
    shift = rp.signal.filters.ShiftFreq(-OFFSET) # The carrier sits OFFSET above the tuned frequency; AFC adjusts this
    front = shift + front[1:] # (channel's own shift is the first item, for a signal at 0)
    assert abs(Fs_mpx - 480e3) < 1, Fs_mpx
    # Audio: both paths end up 15 kHz wide at 48 kS/s, same filters so L+R and L-R stay in time
    def audio_lp():
        lp, Fs_a, _ = rp.signal.channelize(0.0, 2*15e3, Fs_mpx, oversample=1.5) # 480k / 10
        assert abs(Fs_a - AUDIO) < 1, Fs_a # A wrong rate here is a pitch shift and dropouts
        return lp[1:] # Not its ShiftFreq(0): that would make the real audio complex
    tau = args.tau*1e-6
    return SimpleNamespace(
        front=front, shift=shift, demod=rp.wave.FMDemod(gain=-0.5), # FMDemod's sign is opposite the frequency's
        mono_lp=audio_lp(), diff_lp=audio_lp(), pll=PilotPLL(),
        notch_m=rp.signal.filters.IIR(rp.signal.filters.iir.biquad("notch", PILOT/AUDIO, q=8.0)),
        notch_s=rp.signal.filters.IIR(rp.signal.filters.iir.biquad("notch", PILOT/AUDIO, q=8.0)),
        deemph_m=Deemphasis(tau, AUDIO), deemph_s=Deemphasis(tau, AUDIO),
        rds=rds_chain())

def rds_chain():
    """MPX at 480 kS/s -> 1187.5 baud BPSK symbols"""
    front, Fs, D = rp.signal.channelize(57e3, 15e3, 480e3, oversample=1.6) # -> 24 kS/s
    assert abs(Fs - 24e3) < 1, Fs
    pre = front + rp.signal.sample.re.Polyphase(19, 24) + rp.signal.filters.ShapingBiphase() # 19 kS/s = 16 samples per symbol
    sym = rp.sync.time.MM(0.05, 0.0, 32, biphase=True)
    costas = rp.sync.Costas(0.132, 0.00932, order=2)
    return SimpleNamespace(pre=pre, post=sym + costas, sym=sym, costas=costas,
                           track=rp.analysis.quality.Track(np.array([1.0, -1.0])), decoder=RDS())

def reader(blocks, out, stop):
    """The HackRF has to be drained steadily; processing happens on the other side of this queue"""
    try:
        for blk in blocks:
            if stop.is_set():
                break
            try:
                out.put_nowait(blk)
            except queue.Full:
                out.dropped = getattr(out, "dropped", 0) + 1 # Processing can't keep up
    except Exception as e:
        out.put(e)
    out.put(None)

def run(args):
    carrier = acquire(args)
    cf = carrier - OFFSET
    st = build(args, carrier)
    src = rp.source.Device(device(), rx_gain=args.gain, cf=cf, Fs=FS, sps=FS/RDS_BAUD)
    q = queue.Queue(maxsize=24)
    stop = threading.Event()
    threading.Thread(target=reader, args=(src(BLOCK), q, stop), daemon=True).start()
    speaker = Speaker(play=not args.no_audio, wav=args.wav)
    fo, afc, stereo = -OFFSET, 0.0, 0.0 # Shift to the carrier, smoothed carrier error [Hz], stereo blend (0 mono, 1 stereo)
    t0, t_status, busy, nin = time.time(), time.time(), 0.0, 0
    t_mono = time.monotonic()
    print("Running; Ctrl-C to stop")
    try:
        while True:
            blk = q.get()
            if blk is None:
                break
            if isinstance(blk, Exception):
                raise blk
            t1 = time.time()
            nin += len(blk.data)
            mpx = blk + st.front + st.demod
            x = np.asarray(mpx.data)
            # AFC: a carrier error is a DC offset on the demodulated signal. MPX has no DC of its own.
            hz = float(np.mean(x))/(0.5*2*np.pi)*mpx.Fs
            afc += 0.1*(hz - afc)
            if not args.no_afc and abs(afc) > 50:
                fo -= 0.5*afc
                st.shift.fo = fo
                afc *= 0.5

            if not args.no_audio or args.wav:
                # Both paths run the same filters, so L+R and L-R stay in time. De-emphasis is linear: do it before L/R
                m = np.asarray((mpx + st.mono_lp + st.notch_m + st.deemph_m).data)
                d = np.asarray((mpx + st.pll + st.diff_lp + st.notch_s + st.deemph_s).data)
                n = min(len(m), len(d))
                m, d = m[:n], d[:n]
                pll = mpx.state(st.pll)
                pilot_ok = bool(pll and pll.locked)
                stereo += 0.1*((1.0 if pilot_ok and not args.mono else 0.0) - stereo) # Fade, don't click
                speaker.write(args.volume*(m + stereo*d), args.volume*(m - stereo*d))

            if args.rds:
                rb = mpx + st.rds.pre
                out = rb + st.rds.post
                if len(out):
                    st.rds.decoder.bits(np.asarray(out.data).real)
                    st.rds.track.update(out.data, sps=rb.sps)
                for e in st.rds.decoder.events():
                    print(f"  RDS {e}")

            busy += time.time() - t1
            now = time.time()
            if now - t_status >= 2.0:
                pll = mpx.state(st.pll)
                parts = [f"rx {nin/(time.monotonic() - t_mono)/1e6:.3f} MS/s", f"carrier {(cf - fo)/1e6:9.4f} MHz", f"AFC {afc:+6.0f} Hz",
                         f"load {busy/(now - t_status)*100:3.0f}%", f"queue {q.qsize():2d}"]
                if speaker.proc:
                    parts.append(f"buf {speaker.fill*1000:4.0f} ms, {speaker.underruns} underruns, in {speaker.produced/(time.monotonic() - speaker.t_first)*100:6.2f}% of real time")
                if pll:
                    parts.append(f"pilot {pll.level*100:4.1f}% " + ("STEREO" if stereo > 0.5 else "mono  "))
                if args.rds:
                    qual = st.rds.track.update(np.zeros(0, dtype=np.complex64), sps=16)
                    parts.append(str(st.rds.decoder) + (f", {qual}" if qual else ""))
                drop = getattr(q, "dropped", 0)
                if drop:
                    parts.append(f"DROPPED {drop} blocks (too slow)")
                print(" | ".join(parts), flush=True)
                t_status, busy = now, 0.0
            if args.seconds and now - t0 > args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        speaker.close()
    print(f"\n{st.rds.decoder}" if args.rds else "")

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("freq", type=float, nargs="?", help="Station [MHz]")
    ap.add_argument("--scan", action="store_true", help="List the stations the HackRF can hear, then exit")
    ap.add_argument("--rds", action="store_true", help="Decode RDS (station name, text, clock); with --scan, read it from every station")
    ap.add_argument("--rds-time", type=float, default=12.0, metavar="S", help="With --scan --rds: longest to listen to one station (default 12 s)")
    ap.add_argument("--no-audio", action="store_true", help="Don't play the audio")
    ap.add_argument("--mono", action="store_true", help="Never switch to stereo")
    ap.add_argument("--wav", metavar="FILE", help="Also write the audio to a 48 kHz stereo WAV")
    ap.add_argument("--volume", type=float, default=1.8, help="Audio gain (default 1.8: full deviation is about 0.5)")
    ap.add_argument("--tau", type=float, default=75.0, help="De-emphasis [us]: 75 (Americas, Korea), 50 elsewhere")
    ap.add_argument("--no-afc", action="store_true", help="Don't track the carrier")
    ap.add_argument("--seconds", type=float, help="Stop after this long")
    radio_io.add_gain_arg(ap, 0.42)
    args = ap.parse_args()
    if args.scan:
        return scan(args)
    if args.freq is None:
        ap.error("a station frequency is needed (or --scan)")
    run(args)

if __name__ == "__main__":
    sys.exit(main())

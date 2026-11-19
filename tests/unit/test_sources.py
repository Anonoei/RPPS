"""Sources and sinks beyond files and radios: sockets, SigMF recordings, WAV input"""
import json
import socket
import threading

import numpy as np
import pytest

import rpps as rp

pytest.importorskip("pysdrlib")

def _tone(n=10000):
    return (0.5*np.exp(2j*np.pi*0.01*np.arange(n))).astype(np.complex64)

@pytest.mark.parametrize("datatype", ["cf32_le", "ci16_le", "ci8", "cu8", "rf32_be"])
def test_sigmf_round_trip(tmp_path, datatype):
    x = _tone()
    if datatype.startswith("r"):
        x = x.real.copy()
    with rp.source.SigMFSink(tmp_path/"rec", Fs=2e6, cf=915e6, datatype=datatype, annotations=[{"core:sample_start": 10, "core:label": "a"}]) as s:
        for i in range(0, len(x), 3000):
            s.write(x[i:i + 3000])
    meta = json.loads((tmp_path/"rec.sigmf-meta").read_text())
    assert meta["global"]["core:datatype"] == datatype
    src = rp.source.SigMF(tmp_path/"rec.sigmf-data")
    assert (src.Fs, src.cf, len(src)) == (2e6, 915e6, len(x)) and src.annotations[0]["core:label"] == "a"
    blocks = list(src(4096))
    y = np.concatenate([b.data for b in blocks])
    tol = 1e-6 if "f" in datatype else (2/127 if "8" in datatype else 1e-4)
    assert np.allclose(y, x, atol=tol) and blocks[-1].flush and blocks[0].Fs == 2e6

def test_sigmf_rejects_unknown_datatypes():
    with pytest.raises(ValueError):
        rp.source.sigmf.dtype_of("cq16")

def test_wav_in_mono_and_iq(tmp_path):
    a = np.sin(2*np.pi*440*np.arange(4800)/48000)*0.5
    rp.source.write_wav(tmp_path/"a.wav", a, 48000)
    src = rp.source.Wav(tmp_path/"a.wav")
    y = np.concatenate([b.data for b in src(1000)])
    assert src.Fs == 48000 and not src.iq and np.allclose(y, a, atol=1e-4)
    iq = _tone(4800)
    rp.source.write_wav(tmp_path/"iq.wav", np.stack((iq.real, iq.imag), axis=1), 96000)
    y = np.concatenate([b.data for b in rp.source.Wav(tmp_path/"iq.wav")(700)])
    assert np.allclose(y, iq, atol=1e-4)

def _serve_tcp(payload, cuts):
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    def run():
        c, _ = srv.accept()
        i = 0
        for n in cuts: # odd chunk sizes: samples split across sends
            c.sendall(payload[i:i + n])
            i += n
        c.sendall(payload[i:])
        c.close()
        srv.close()
    threading.Thread(target=run, daemon=True).start()
    return srv.getsockname()[1]

def test_tcp_source_whole_samples_whatever_the_cut():
    x = _tone(5000)
    port = _serve_tcp(x.tobytes(), [3, 1001, 7, 4096])
    blocks = list(rp.source.Socket("cf64", f"tcp://127.0.0.1:{port}", Fs=1e6)(1024))
    y = np.concatenate([b.data for b in blocks])
    assert np.array_equal(y, x) and [len(b) for b in blocks[:-1]] == [1024]*4 and blocks[-1].flush

def test_udp_source_ends_on_silence():
    src = rp.source.Socket("ci16t", "udp://127.0.0.1:0", Fs=1e6, timeout=0.5)
    s = src._open()
    port = s.getsockname()[1]
    s.close()
    src = rp.source.Socket("ci16t", f"udp://127.0.0.1:{port}", Fs=1e6, timeout=0.5)
    raw = (np.round(_tone(2000).view(np.float32)*127)).astype(np.int8).tobytes()
    def send():
        import time
        time.sleep(0.2)
        tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for i in range(0, len(raw), 1000):
            tx.sendto(raw[i:i + 1000], ("127.0.0.1", port))
        tx.close()
    threading.Thread(target=send, daemon=True).start()
    y = np.concatenate([b.data for b in src(512)])
    assert len(y) == 2000 and np.allclose(y, _tone(2000), atol=2/127)

def test_bad_urls():
    with pytest.raises(ValueError):
        rp.source.Socket("cf64", "http://x:1")

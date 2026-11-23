"""Batch kernels of the code operators: `BlockDecode` of a block code or interleaver, `XorMask` of an additive scrambler

Checked row for row against the operator run frame by frame (tests/unit/test_batch_kernels.py); they use only what
numpy and cupy share, so `xp` can be either (rpps.parallel)
"""
import numpy as np

from ..base.data import Digital, Mod
from ..base.kernel import Form, Kernel, device_functions, has_device_kernels
from ..base.types import dtype, Rep
from ..errors import RPPSError


class BlockDecode(Kernel):
    """A decoder or deinterleaver of whole blocks, over a batch: the frames' blocks go through the coder in one call (a block code's
    decoder already runs its codewords on several threads, so a batch is more of them at once, not more calls). Each block is
    independent of the others, so a row comes out as it would alone. A frame is whole blocks; items past the last block are
    dropped and reported, as the operator does at the end of a flushed stream.

    coder: what the operator decodes with (`decode(bits or LLRs) -> bits or LLRs`, any number of blocks); block: items a
    block takes in; soft: it takes LLRs when it is given them"""
    def __init__(self, coder, block, soft, category, rep, name="block"):
        self.coder, self.block, self.soft, self.category, self.rep = coder, int(block), bool(soft), category, rep
        self.name = f"decode:{name}"

    where = "host" # (the decoders are C++ and numpy; a device decoder is a kernel of its own)

    def shape(self, n_in):
        return None # a code says how many items a block gives, not every coder how its input and output relate

    def _in(self, b, X):
        f = b.form
        if f.dt == dtype.LLR:
            return X if (self.soft and issubclass(f.cls, Mod)) else X < 0
        return X.astype(bool)

    def run(self, xp, b):
        N, L = b.x.shape
        used = L//self.block*self.block
        for i in range(N):
            if used < L and not b.bad[i]:
                b.notes.setdefault(i, []).append({"what": "cut short", "items": int(L - used), "at": int(b.pos[i] + b.n_in)})
        if used == 0:
            b.x = np.zeros((N, 0), dtype=bool)
            b.form = Form(Digital, self.category, dtype.BITS, self.rep, *b.form[4:])
            return b
        x = self._in(b, b.x[:, :used])
        try:
            y, errs = self.coder.decode(x.reshape(-1), status=True)
            y = np.asarray(y).reshape(N, -1)
        except RPPSError: # one frame's block was bad: find which, the others go on
            rows = []
            for i in range(N):
                try:
                    rows.append(np.asarray(self.coder.decode(x[i])).reshape(-1))
                except RPPSError as e:
                    b.fail(i, f"{type(e).__name__}: {e}")
                    rows.append(None)
            ok = next((r for r in rows if r is not None), None)
            if ok is None:
                b.x = np.zeros((N, 0), dtype=bool)
                return b
            y = np.zeros((N, len(ok)), dtype=ok.dtype)
            for i, r in enumerate(rows):
                if r is not None:
                    y[i] = r
            errs = None # (decoded row by row: no verdict per codeword)
        soft = y.dtype.kind == "f"
        b.x = y
        b.form = Form(Mod if soft else Digital, self.category, dtype.LLR if soft else dtype.BITS, self.rep, *b.form[4:])
        if errs is not None and len(errs) and len(errs) % N == 0: # This decoder's verdict per row (the FAILED mask, per frame)
            b.code_failed = (np.asarray(errs).reshape(N, -1) < 0).any(axis=1)
        return b

class XorMask(Kernel):
    """Bits XOR a fixed mask repeated from the start of each frame (a scrambler restarted every `len(mask)` bits). LLRs
    stay LLRs, their signs flipped (as the additive scrambler descrambles them)"""
    def __init__(self, mask, category, rep, name="xor"):
        self.mask, self.category, self.rep = np.asarray(mask, dtype=bool), category, rep
        self.name = f"xor:{name}"

    def shape(self, n_in):
        return n_in

    def consts(self):
        return {"mask": self.mask}

    def run(self, xp, b):
        L = b.x.shape[1]
        m = self.const(xp, L, lambda: self.mask[np.arange(L) % len(self.mask)])[None, :]
        if b.form.dt == dtype.LLR and self.rep == Rep.DATA: # Descrambling (encoding decides first, as the operator does)
            b.x = (b.x*(1.0 - 2.0*m)).astype(b.x.dtype)
            b.form = Form(Mod, self.category, dtype.LLR, self.rep, *b.form[4:])
        else:
            b.x = (b.x < 0 if b.form.dt == dtype.LLR else b.x.astype(bool)) ^ m
            b.form = Form(Digital, self.category, dtype.BITS, self.rep, *b.form[4:])
        return b

# Layered normalized min-sum on a device, as `ldpc.layered` and the C++ ldpc_minsum: one launch per layer over every
# (check, codeword) of it (a layer's checks share no variable, so they write without races), then a parity launch. The
# codewords are the fastest axis (total [N, B], r [E, B]): neighbouring threads are one check of neighbouring codewords, so
# their reads and writes are contiguous. A portable subset of CUDA C that hiprtc compiles as it is: no warp intrinsics, no
# vendor headers. float32 (consumer GPUs run float64 slowly); the decisions match the float64 host decoder away from ties.
LDPC_SOURCE = r"""
#define LDPC_INF 3.402823466e+38f
extern "C" __global__ void ldpc_layer(float *total, float *r, const int *var, const int *cstart, const int *deg,
                                      const int *rows, const int r0, const int nrows, const int N, const int E,
                                      const float alpha, const int *done, const int B) {
    const long long t = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= (long long)nrows*B) return;
    const int b = (int)(t % B), c = rows[r0 + (int)(t/B)];
    if (done[b]) return;
    const int e0 = cstart[c], d = deg[c];
    float m1 = LDPC_INF, m2 = LDPC_INF;
    int i1 = 0, neg = 0;
    for (int j = 0; j < d; j++) {
        const float v = total[(long long)var[e0 + j]*B + b] - r[(long long)(e0 + j)*B + b], a = fabsf(v);
        neg ^= v < 0.0f;
        if (j == 0 || a < m1) { m2 = m1; m1 = a; i1 = j; } // (the first minimum, as argmin)
        else if (a < m2) m2 = a;
    }
    for (int j = 0; j < d; j++) {
        const long long ti = (long long)var[e0 + j]*B + b, ri = (long long)(e0 + j)*B + b;
        const float v = total[ti] - r[ri];
        const float s = ((neg ^ (v < 0.0f)) ? -alpha : alpha);
        const float nr = s*(j == i1 ? m2 : m1);
        r[ri] = nr;
        total[ti] = v + nr;
    }
}
extern "C" __global__ void ldpc_parity(const float *total, const int *var, const int *cstart, const int *deg, const int M,
                                       const int N, int *fail, const int *done, const int B) {
    const long long t = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= (long long)M*B) return;
    const int b = (int)(t % B), c = (int)(t/B);
    if (done[b]) return;
    int par = 0;
    for (int j = 0; j < deg[c]; j++) par ^= total[(long long)var[cstart[c] + j]*B + b] < 0.0f;
    if (par) fail[b] = 1;
}
"""

class LdpcDecode(BlockDecode):
    """`BlockDecode` of a layered min-sum LDPC code that decodes on the device it is given: the batch's codewords in one
    run of launches, the graph uploaded once. On numpy (and on an array module without kernels of its own) it is
    `BlockDecode`, the host decoder: the oracle. Sum-product (`check` at the effort level) decodes on the host"""
    THREADS = 256

    def __init__(self, coder, block, soft, category, rep, name="block"):
        super().__init__(coder, block, soft, category, rep, name)
        self.name = f"decode:{name}@device"

    where = "any"

    def consts(self):
        c = self.coder
        return {"var": c.var, "cstart": c.cstart, "deg": c.deg, "rows": np.concatenate(c.layers()), "info": c.info}

    def _host(self, xp, b):
        """The host decoder, with the batch moved explicitly (an array module without kernels of its own, sum-product)"""
        from ..base.kernel import to_host
        b.x = to_host(b.x)
        b = super().run(np, b)
        b.x = xp.asarray(b.x)
        return b

    def run(self, xp, b):
        from .. import _effort
        if xp is np:
            return super().run(np, b)
        c = self.coder
        if not has_device_kernels(xp) or _effort.choice("check", c.check) == "sumproduct":
            return self._host(xp, b)
        R, L = b.x.shape
        n = c.N
        used = L//n*n
        for i in range(R):
            if used < L and not b.bad[i]:
                b.notes.setdefault(i, []).append({"what": "cut short", "items": int(L - used), "at": int(b.pos[i] + b.n_in)})
        if used == 0:
            b.x = xp.zeros((R, 0), dtype=bool)
            b.form = Form(Digital, self.category, dtype.BITS, self.rep, *b.form[4:])
            return b
        x = b.x[:, :used]
        if b.form.dt == dtype.LLR and self.soft and issubclass(b.form.cls, Mod):
            llr = x.astype(xp.float32)
        else: # hard bits as +-2, as as_llr
            bits = (x < 0) if b.form.dt == dtype.LLR else x.astype(bool)
            llr = xp.float32(2.0)*(xp.float32(1.0) - xp.float32(2.0)*bits.astype(xp.float32))
        total = xp.ascontiguousarray(llr.reshape(-1, n).T) # (N, codewords): the codewords fastest
        Bw = total.shape[1]
        f = device_functions(xp, LDPC_SOURCE, ("ldpc_layer", "ldpc_parity"))
        layer, parity = f["ldpc_layer"], f["ldpc_parity"]
        i32 = lambda a: np.ascontiguousarray(a, dtype=np.int32)
        var = self.const(xp, "var", lambda: i32(c.var))
        cstart = self.const(xp, "cstart", lambda: i32(c.cstart))
        deg = self.const(xp, "deg", lambda: i32(c.deg))
        layers = c.layers()
        rows = self.const(xp, "rows", lambda: i32(np.concatenate(layers)))
        info = self.const(xp, "info", lambda: np.asarray(c.info, dtype=np.int64))
        bounds = np.cumsum([0] + [len(l) for l in layers])
        E, M = len(c.var), c.M
        # The codewords still decoding, compacted as others converge: a wavefront spans neighbouring codewords, so one that
        # is done would still cost a whole iteration while any beside it is not
        cols = np.arange(Bw)
        tw, rw = total, xp.zeros((E, Bw), dtype=xp.float32)
        live = xp.zeros(Bw, dtype=xp.int32) # (the kernels' done flags: every working codeword is live)
        fail = xp.zeros(Bw, dtype=xp.int32)
        conv = np.zeros(Bw, dtype=bool)
        T, alpha = self.THREADS, np.float32(c.alpha)
        for _ in range(_effort.scale("iters", c.iters)):
            W = len(cols)
            for k in range(len(layers)):
                nr = int(bounds[k + 1] - bounds[k])
                layer(((nr*W + T - 1)//T,), (T,), (tw, rw, var, cstart, deg, rows, np.int32(bounds[k]), np.int32(nr),
                                                   np.int32(n), np.int32(E), alpha, live, np.int32(W)))
            fail[:W].fill(0)
            parity(((M*W + T - 1)//T,), (T,), (tw, var, cstart, deg, np.int32(M), np.int32(n), fail, live, np.int32(W)))
            ok = fail[:W].get() == 0 # (one small read per iteration)
            if not ok.any():
                continue
            conv[cols[ok]] = True
            if ok.all():
                if tw is not total:
                    total[:, xp.asarray(cols)] = tw
                cols = cols[:0]
                break
            keep = xp.asarray(np.nonzero(~ok)[0])
            if tw is not total:
                total[:, xp.asarray(cols[ok])] = tw[:, xp.asarray(np.nonzero(ok)[0])]
            tw, rw, cols = xp.ascontiguousarray(tw[:, keep]), xp.ascontiguousarray(rw[:, keep]), cols[~ok]
        if len(cols) and tw is not total:
            total[:, xp.asarray(cols)] = tw
        y = (total[info] < 0).T
        b.x = y.reshape(R, -1)
        b.form = Form(Digital, self.category, dtype.BITS, self.rep, *b.form[4:])
        b.code_failed = (~conv).reshape(R, -1).any(axis=1)
        return b

class Identity(Kernel):
    """A code or interleaver that a frame's parameters switch off (`control(name=None)`): the items pass as they are, now
    the code's category, as the operator gives them"""
    def __init__(self, category, name="none"):
        self.category = category
        self.name = f"pass:{name}"

    def run(self, xp, b):
        b.form = Form(b.form.cls, self.category, b.form.dt, b.form.rep, *b.form[4:])
        return b

class Permute(BlockDecode):
    """`BlockDecode` of a block interleaver that is a permutation: every block of a row gathered by one index, on whichever
    `xp` it is given (the index goes to a device once). `perm`: what the deinterleaver gives for the input positions
    0 .. block-1"""
    def __init__(self, perm, coder, block, soft, category, rep, name="block"):
        super().__init__(coder, block, soft, category, rep, name)
        self.perm = np.asarray(perm, dtype=np.int64)
        self.name = f"permute:{name}"

    where = "any"

    def shape(self, n_in):
        return None if n_in is None else n_in//self.block*self.block

    def consts(self):
        return {"perm": self.perm}

    def run(self, xp, b):
        R, L = b.x.shape
        used = L//self.block*self.block
        for i in range(R):
            if used < L and not b.bad[i]:
                b.notes.setdefault(i, []).append({"what": "cut short", "items": int(L - used), "at": int(b.pos[i] + b.n_in)})
        idx = self.const(xp, ("perm", used), lambda: (np.arange(0, used, self.block)[:, None] + self.perm[None, :]).reshape(-1))
        x = self._in(b, b.x[:, :used])
        b.x = x[:, idx]
        soft = b.x.dtype.kind == "f"
        b.form = Form(Mod if soft else Digital, self.category, dtype.LLR if soft else dtype.BITS, self.rep, *b.form[4:])
        return b

def permutation_of(coder, block):
    """The index a block deinterleaver gathers with (decoding positions 0 .. block-1), or None when it is not a pure
    permutation (it changes values, or loses some)"""
    try:
        y = np.asarray(coder.decode(np.arange(block, dtype=np.float64)))
    except Exception: # (a decoder that takes only bits)
        return None
    if y.shape != (block,) or not np.array_equal(np.sort(y), np.arange(block)):
        return None
    return y.astype(np.int64)

# A binary BCH code's syndromes on a device, as rpps._native gf_syndromes (fcr = prim = 1): a block per (word, root),
# its threads over the word's bits, S_j = XOR of alpha^(j (n-1-k)) over the bits k that are set, reduced in shared memory
# (a power-of-two block). Portable CUDA C
BCH_SOURCE = r"""
extern "C" __global__ void bch_syndromes(const unsigned char *words, const int n, const int *exp_, const int q,
                                         const int nroots, int *S) {
    extern __shared__ int part[];
    const int w = blockIdx.x, j = blockIdx.y + 1;
    const unsigned char *x = words + (long long)w*n;
    int acc = 0;
    for (int k = threadIdx.x; k < n; k += blockDim.x)
        if (x[k]) acc ^= exp_[(int)(((long long)j*(n - 1 - k)) % q)];
    part[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) part[threadIdx.x] ^= part[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) S[(long long)w*nroots + blockIdx.y] = part[0];
}
"""

class BchDecode(BlockDecode):
    """`BlockDecode` of a binary BCH code (`bchgf`) that finds the errors on the device it is given: every codeword's
    syndromes in one launch; only the words that have errors (few, after an LDPC code) come to the host to be located
    (Berlekamp-Massey and Chien, as the host decoder), and their bits are flipped on the device. On numpy, and on an array
    module without kernels of its own, it is `BlockDecode`"""
    THREADS = 256

    def __init__(self, coder, block, soft, category, rep, name="block"):
        super().__init__(coder, block, soft, category, rep, name)
        self.name = f"decode:{name}@device"

    where = "any"

    def _host(self, xp, b):
        from ..base.kernel import to_host
        b.x = to_host(b.x)
        b = super().run(np, b)
        b.x = xp.asarray(b.x)
        return b

    def run(self, xp, b):
        if xp is np:
            return super().run(np, b)
        if not has_device_kernels(xp):
            return self._host(xp, b)
        c = self.coder
        R, L = b.x.shape
        n = c.den
        used = L//n*n
        for i in range(R):
            if used < L and not b.bad[i]:
                b.notes.setdefault(i, []).append({"what": "cut short", "items": int(L - used), "at": int(b.pos[i] + b.n_in)})
        if used == 0:
            b.x = xp.zeros((R, 0), dtype=bool)
            b.form = Form(Digital, self.category, dtype.BITS, self.rep, *b.form[4:])
            return b
        x = b.x[:, :used]
        x = (x < 0) if b.form.dt == dtype.LLR else x.astype(bool)
        words = xp.ascontiguousarray(x.reshape(-1, n)) # (W, n), one byte a bit
        W, nroots, q = words.shape[0], 2*c.t, c.size - 1
        exp_ = self.const(xp, "exp", lambda: np.asarray(c.exp[:q], dtype=np.int32))
        S = xp.empty((W, nroots), dtype=xp.int32)
        f = device_functions(xp, BCH_SOURCE, ("bch_syndromes",))["bch_syndromes"]
        T = self.THREADS
        f((W, nroots), (T,), (words.view(xp.uint8), np.int32(n), exp_, np.int32(q), np.int32(nroots), S), shared_mem=4*T)
        errs = np.zeros(W, dtype=int)
        bad = np.nonzero(S.any(axis=1).get())[0] # (a word's errors are located on the host: few, and a short serial search)
        if len(bad):
            Sb = S[xp.asarray(bad)].get().astype(np.int64)
            rows, cols = [], []
            for w, s in zip(bad, Sb):
                pos = c._locate(s)
                if pos is None:
                    errs[w] = -1
                else:
                    errs[w] = len(pos)
                    rows.extend([w]*len(pos))
                    cols.extend(int(p) for p in pos)
            if rows:
                r, k = xp.asarray(np.array(rows, dtype=np.int64)), xp.asarray(np.array(cols, dtype=np.int64))
                words[r, k] = ~words[r, k]
        b.x = words[:, :c.num].reshape(R, -1)
        b.form = Form(Digital, self.category, dtype.BITS, self.rep, *b.form[4:])
        if len(errs) % R == 0:
            b.code_failed = (errs.reshape(R, -1) < 0).any(axis=1)
        return b

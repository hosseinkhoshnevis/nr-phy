"""The 5G NR LDPC of TS 38.212, here in its native habitat: base
graph 1, lifting Z = 384, K = 8448, with the standard's own rate
matching. The encoder and min-sum decoder are shared with the
author's free-space package, which borrowed this code from 5G; the
circular buffer, redundancy-version start, and the Q_m-row bit
interleaver of clause 5.4.2 are what this package adds.

Rate matching reads E bits from the circular buffer d (the codeword
minus its first two systematic blocks, N_cb = 66 Z), wrapping around
for rates below 11/68, then interleaves them across the modulation
levels: f[i Q_m + k] = e[k E/Q_m + i], which spreads each code bit
stream over the reliability levels of the constellation. Rate
recovery inverts both and adds repeated LLRs where the buffer
wrapped.
"""

import numpy as np

from .bg1 import SHIFTS, Z

N_ROWS = len(SHIFTS)          # 46
N_COLS = len(SHIFTS[0])       # 68
K_BLOCKS = 22
K_BITS = K_BLOCKS * Z         # 8448
N_CB = 66 * Z                 # circular buffer length

EDGES = [(i, j, s) for i, row in enumerate(SHIFTS)
         for j, s in enumerate(row) if s >= 0]
_E_ROW = np.array([e[0] for e in EDGES])
_E_COL = np.array([e[1] for e in EDGES])
_E_SHIFT = np.array([e[2] for e in EDGES])


def _mul(shift, x):
    return np.roll(x, -shift, axis=-2)


def encode(info):
    """info (B, 8448) -> full codeword (B, 68*384), first 2Z punctured
    later by rate matching."""
    info = np.asarray(info, dtype=np.uint8)
    B = info.shape[0]
    u = info.reshape(B, K_BLOCKS, Z).transpose(1, 2, 0)
    lam = np.zeros((4, Z, B), dtype=np.uint8)
    for i in range(4):
        for j in range(K_BLOCKS):
            s = SHIFTS[i][j]
            if s >= 0:
                lam[i] ^= _mul(s, u[j])
    s22 = [SHIFTS[i][22] for i in range(4) if SHIFTS[i][22] >= 0]
    surv = [s for s in set(s22) if s22.count(s) % 2 == 1]
    q = np.zeros((4, Z, B), dtype=np.uint8)
    q[0] = np.roll(lam[0] ^ lam[1] ^ lam[2] ^ lam[3], surv[0], axis=0)
    known = [True, False, False, False]
    for _ in range(3):
        for i in range(4):
            terms, unknown = lam[i].copy(), None
            for c in (22, 23, 24, 25):
                s = SHIFTS[i][c]
                if s < 0:
                    continue
                if known[c - 22]:
                    terms ^= _mul(s, q[c - 22])
                elif unknown is None:
                    unknown = (c - 22, s)
                else:
                    unknown = "two"
            if unknown not in (None, "two"):
                idx, s = unknown
                q[idx] = np.roll(terms, s, axis=0)
                known[idx] = True
    cols = np.zeros((N_COLS, Z, B), dtype=np.uint8)
    cols[:K_BLOCKS] = u
    cols[22:26] = q
    for i in range(4, N_ROWS):
        acc = np.zeros((Z, B), dtype=np.uint8)
        for j in range(26):
            s = SHIFTS[i][j]
            if s >= 0:
                acc ^= _mul(s, cols[j])
        cols[26 + (i - 4)] = acc
    return cols.transpose(2, 0, 1).reshape(B, N_COLS * Z)


def check(codeword):
    c = np.asarray(codeword, dtype=np.uint8)
    cols = c.reshape(c.shape[0], N_COLS, Z).transpose(1, 2, 0)
    bad = 0
    for i in range(N_ROWS):
        acc = np.zeros(cols.shape[1:], dtype=np.uint8)
        for j in range(N_COLS):
            s = SHIFTS[i][j]
            if s >= 0:
                acc ^= _mul(s, cols[j])
        bad += int(acc.sum())
    return bad


def rate_match(codeword, e_bits, qm):
    """Codeword (B, 68Z) -> transmitted bits (B, E), rv0, with the
    clause 5.4.2.2 bit interleaver. E must divide by Qm."""
    assert e_bits % qm == 0
    d = codeword[:, 2 * Z:]                       # circular buffer, 66Z
    idx = np.arange(e_bits) % N_CB
    e = d[:, idx]
    rows = e.reshape(-1, qm, e_bits // qm)        # f[i*Qm+k] = e[k*E/Qm+i]
    return rows.transpose(0, 2, 1).reshape(-1, e_bits)


def rate_recover(llr_f, e_bits, qm):
    """Received LLRs (B, E) -> circular-buffer LLRs (B, 66Z), repeats
    combined; punctured systematic blocks are the caller's zeros."""
    B = llr_f.shape[0]
    e = llr_f.reshape(B, e_bits // qm, qm).transpose(0, 2, 1).reshape(B, e_bits)
    buf = np.zeros((B, N_CB), dtype=np.float32)
    idx = np.arange(e_bits) % N_CB
    np.add.at(buf, (slice(None), idx), e.astype(np.float32))
    return buf


def parity_blocks_used(e_bits):
    """How many parity blocks the decoder should work with."""
    nb = int(np.ceil(max(e_bits - 20 * Z, 4 * Z) / Z))
    return int(np.clip(nb, 4, 46))


class Decoder:
    """Normalised min-sum (flooding, alpha 0.75) on the sub-graph the
    transmitted E actually pays for, exactly as in the free-space
    sibling; the punctured systematic blocks enter at zero LLR and are
    recovered by the iteration."""

    def __init__(self, e_bits, iters=40, alpha=0.75):
        self.nb = parity_blocks_used(e_bits)
        self.e_bits = e_bits
        self.n_cols = K_BLOCKS + self.nb
        self.iters = iters
        self.alpha = alpha
        m = (_E_ROW < self.nb) & (_E_COL < self.n_cols)
        self.row = _E_ROW[m]
        self.col = _E_COL[m]
        self.shift = _E_SHIFT[m]
        self.ptr = np.searchsorted(self.row, np.arange(self.nb + 1))

    def decode(self, llr_buf):
        """Circular-buffer LLRs (B, 66Z) -> (info (B, 8448), ok)."""
        B = llr_buf.shape[0]
        cols = np.zeros((self.n_cols, Z, B), dtype=np.float32)
        used = (self.n_cols - 2) * Z
        cols[2:] = llr_buf[:, :used].reshape(B, self.n_cols - 2, Z).transpose(1, 2, 0)
        E = len(self.row)
        c2v = np.zeros((E, Z, B), dtype=np.float32)
        post = cols.copy()
        for it in range(self.iters):
            v2c = post[self.col] - c2v
            chk = np.empty_like(v2c)
            for e in range(E):
                chk[e] = np.roll(v2c[e], -self.shift[e], axis=0)
            sgn = np.where(chk < 0, -1.0, 1.0).astype(np.float32)
            mag = np.abs(chk)
            new = np.empty_like(chk)
            for r in range(self.nb):
                a, b = self.ptr[r], self.ptr[r + 1]
                s_all = np.prod(sgn[a:b], axis=0)
                m = mag[a:b]
                i1 = np.argmin(m, axis=0)
                m1 = np.take_along_axis(m, i1[None], axis=0)[0]
                m_masked = m.copy()
                np.put_along_axis(m_masked, i1[None], np.inf, axis=0)
                m2 = m_masked.min(axis=0)
                mins = np.where(np.arange(b - a)[:, None, None] == i1[None],
                                m2[None], m1[None])
                new[a:b] = (s_all[None] * sgn[a:b]) * (self.alpha * mins)
            for e in range(E):
                c2v[e] = np.roll(new[e], self.shift[e], axis=0)
            post = cols.copy()
            np.add.at(post, self.col, c2v)
            if it % 3 == 2 or it == self.iters - 1:
                hard = (post < 0).astype(np.uint8)
                if self._syndrome_ok(hard).all():
                    break
        hard = (post < 0).astype(np.uint8)
        ok = self._syndrome_ok(hard)
        info = hard[:K_BLOCKS].transpose(2, 0, 1).reshape(B, K_BITS)
        return info, ok

    def _syndrome_ok(self, cols_hard):
        B = cols_hard.shape[2]
        bad = np.zeros(B, dtype=np.int64)
        for r in range(self.nb):
            a, b = self.ptr[r], self.ptr[r + 1]
            acc = np.zeros((Z, B), dtype=np.uint8)
            for e in range(a, b):
                acc ^= np.roll(cols_hard[self.col[e]], -self.shift[e], axis=0)
            bad += acc.sum(axis=0).astype(np.int64)
        return bad == 0

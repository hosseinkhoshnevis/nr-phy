"""QAM mapping and soft demapping per TS 38.211 clause 5.1.

The standard's constellations are Gray squares built from a per-axis
PAM recursion, which is what makes both the mapper and the max-log
demapper factor into two independent real problems per subcarrier:
even-indexed bits drive I, odd-indexed bits drive Q. LLRs take a
per-RE noise variance, because after a fading channel and a one-tap
equalizer every subcarrier lives at its own SNR.
"""

import numpy as np

_NORM = {4: 2.0, 16: 10.0, 64: 42.0, 256: 170.0}


def pam_levels(bits_per_axis):
    """Level for every bit pattern of one axis, 38.211's nested rule:
    (1-2b0) * (2^{m-1} - (1-2b1)(2^{m-2} - ... (2 - (1-2b_{m-1}))))."""
    m = bits_per_axis
    if m == 1:
        return np.array([1.0, -1.0])
    out = np.zeros(1 << m)
    for pattern in range(1 << m):
        bits = [(pattern >> (m - 1 - k)) & 1 for k in range(m)]
        val = 1.0
        for k in range(m - 1, 0, -1):
            val = (2 ** (m - k)) - (1 - 2 * bits[k]) * val
        out[pattern] = (1 - 2 * bits[0]) * val
    return out


def qam_map(bits, qam):
    """bits (B, n), n divisible by log2(qam) -> symbols (B, n/Qm)."""
    qm = int(np.log2(qam))
    ax = qm // 2
    lv = pam_levels(ax)
    b = bits.reshape(bits.shape[0], -1, qm)
    weights = 1 << np.arange(ax - 1, -1, -1)
    i_idx = (b[:, :, 0::2] * weights).sum(axis=2)
    q_idx = (b[:, :, 1::2] * weights).sum(axis=2)
    return (lv[i_idx] + 1j * lv[q_idx]) / np.sqrt(_NORM[qam])


def qam_llrs(symbols, qam, noise_var):
    """Max-log LLRs in 38.211 bit order, positive = bit 0. noise_var
    broadcasts against `symbols`, one value per resource element."""
    qm = int(np.log2(qam))
    ax = qm // 2
    lv = pam_levels(ax) / np.sqrt(_NORM[qam])
    nv = np.broadcast_to(np.maximum(noise_var, 1e-12), symbols.shape)
    out = np.empty(symbols.shape + (qm,), dtype=np.float32)
    for axis, u in ((0, symbols.real), (1, symbols.imag)):
        d2 = (u[..., None] - lv) ** 2
        for bit in range(ax):
            mask0 = np.array([(p >> (ax - 1 - bit)) & 1 == 0
                              for p in range(1 << ax)])
            m0 = d2[..., mask0].min(axis=-1)
            m1 = d2[..., ~mask0].min(axis=-1)
            out[..., 2 * bit + axis] = (m1 - m0) / nv
    return np.clip(out.reshape(symbols.shape[:-1] + (-1,)), -60, 60)


def exact_ber_qam_awgn(qam, esn0_db):
    """Exact uncoded Gray-QAM bit error rate on AWGN, computed from
    the PAM decision regions rather than from an approximation: the
    theory anchor of the whole chain."""
    from math import erfc, sqrt
    qm = int(np.log2(qam))
    ax = qm // 2
    lv = pam_levels(ax) / np.sqrt(_NORM[qam])
    order = np.argsort(lv)
    sigma = sqrt(1.0 / (2.0 * 10 ** (esn0_db / 10)))   # per axis
    edges = 0.5 * (lv[order][1:] + lv[order][:-1])
    total = 0.0
    n = 1 << ax
    for pat_tx in range(n):
        x = lv[pat_tx]
        for pos in range(n):
            pat_rx = order[pos]
            lo = -np.inf if pos == 0 else edges[pos - 1]
            hi = np.inf if pos == n - 1 else edges[pos]
            p = 0.5 * (erfc((lo - x) / (sqrt(2) * sigma))
                       - erfc((hi - x) / (sqrt(2) * sigma)))
            if pat_rx != pat_tx:
                total += p * bin(int(pat_rx) ^ pat_tx).count("1")
    return total / (n * ax)


def bicm_capacity(qam, esn0_db, n_grid=4000):
    """Per-axis numerical BICM mutual information, bits per symbol.
    Gray square QAM factors into two PAM axes, so two 1-D integrals
    per bit level give the soft-decision ceiling any code sees."""
    qm = int(np.log2(qam))
    ax = qm // 2
    lv = pam_levels(ax) / np.sqrt(_NORM[qam])
    sigma = np.sqrt(1.0 / (2.0 * 10 ** (esn0_db / 10)))
    u = np.linspace(lv.min() - 6 * sigma, lv.max() + 6 * sigma, n_grid)
    du = u[1] - u[0]
    pdf = np.exp(-(u[:, None] - lv) ** 2 / (2 * sigma ** 2)) \
        / np.sqrt(2 * np.pi * sigma ** 2)              # (grid, levels)
    n = 1 << ax
    mi = 0.0
    for bit in range(ax):
        b_of = np.array([(p >> (ax - 1 - bit)) & 1 for p in range(n)])
        p_all = pdf.mean(axis=1)
        for val in (0, 1):
            p_cond = pdf[:, b_of == val].mean(axis=1)
            good = p_cond > 1e-300
            mi += 0.5 * np.sum(p_cond[good] * np.log2(
                p_cond[good] / np.maximum(p_all[good], 1e-300))) * du
    return 2.0 * mi                                    # both axes

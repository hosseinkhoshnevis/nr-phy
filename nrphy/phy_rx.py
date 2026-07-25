"""Receive chain: front end, blind synchronisation, FFT, channel
estimation, the one-tap equalizer, and per-RE noise for the demapper.

Sync is earned, not assumed. Timing and fractional CFO come from the
cyclic prefix itself (van de Beek's ML estimator: every OFDM symbol
repeats its own tail, so correlating r with r delayed by N lights up
at the CP of every symbol); the slot boundary and integer CFO come
from the DM-RS, which is the one sequence a receiver knows in
advance. Channel estimation is least squares at the pilots with a
DFT-domain cleanup: the true channel lives inside the CP length, so
everything the LS estimate shows beyond it is noise and is thrown
away, which is worth several dB of estimation SNR.
"""

import numpy as np

from .grid import cp_lengths, data_re_map, dmrs_sequence, slot_samples
from .mod import qam_llrs


def rx_frontend(y, cfg, rng):
    """I/Q demodulator imbalance, AGC, ADC, decimation to the base
    rate, and the AWGN that sets Es/N0. The noise is added in-band at
    the base rate against the measured signal power, so the configured
    SNR per occupied subcarrier is exact by construction rather than
    tracked through the chain's scalings."""
    if cfg.include_impairments:
        g = 10 ** (cfg.rx_iq_amp_db / 20)
        phi = np.deg2rad(cfg.rx_iq_phase_deg)
        y = y.real + 1j * g * (y.imag * np.cos(phi) + y.real * np.sin(phi))
    y = y / np.sqrt(np.mean(np.abs(y) ** 2))       # AGC to unit power
    if cfg.adc_bits and cfg.include_impairments:
        from .phy_tx import _quantise
        y = _quantise(y, cfg.adc_bits)
    spec = np.fft.fft(y)
    n = y.size // cfg.os
    down = np.concatenate([spec[:n // 2], spec[-n // 2:]])
    yb = np.fft.ifft(down) / cfg.os
    p_b = float(np.mean(np.abs(yb) ** 2))
    es = p_b * cfg.n_fft / cfg.n_sc
    sigma2 = es / 10 ** (cfg.snr_db / 10)
    noise = (rng.standard_normal(yb.size) + 1j * rng.standard_normal(yb.size))
    return yb + noise * np.sqrt(sigma2 / 2.0)


def symbol_offsets(cfg, n=14):
    """Where each symbol of a slot starts, in samples from the slot
    boundary. The first prefix is the long one, which is why a slot is
    not simply 14 equal symbols and why folding at the symbol period
    drifts."""
    cp0, cp = cp_lengths(cfg)
    lens = [(cp0 if l == 0 else cp) + cfg.n_fft for l in range(n)]
    return np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(int)


def cp_sync(y, cfg, n_symbols):
    """Van de Beek timing and fractional CFO from the cyclic prefix,
    folded at the slot period.

    Every OFDM symbol ends with a copy of its own head, so the product
    r[j] r*[j+N] is coherent for as long as the correlation window
    stays inside a prefix; its magnitude says where the symbol starts
    and its phase says how far the carrier has drifted, since N
    samples of offset are one full subcarrier spacing of phase. The
    fold is at the slot, not at the symbol: the first prefix of every
    slot is longer than the rest, so a symbol-period fold slides by
    those extra samples once per slot and drags the peak late by more
    and more as the capture grows. Summing the 14 prefixes of a slot
    at their true offsets keeps every capture length honest. The
    windowed part of each prefix is skipped, so the WOLA taper cannot
    pull the peak either."""
    n = cfg.n_fft
    cp0, cp = cp_lengths(cfg)
    w = min(cfg.window_samples, cp - 8)
    cw = cp - w                                   # clean part of a CP
    lag = y[:-n] * np.conj(y[n:])
    acc = np.convolve(lag, np.ones(cw), mode="valid")
    slot = slot_samples(cfg)
    offs = symbol_offsets(cfg) + w
    n_folds = max(int((acc.size - offs[-1]) // slot), 1)
    m = np.zeros(slot, dtype=complex)
    for s in range(n_folds):
        for o in offs:
            base = s * slot + o
            seg = acc[base:base + slot]
            m[:seg.size] += seg
    tau = int(np.argmax(np.abs(m)))
    # then back the estimate off into the prefix on purpose: the two
    # directions are not symmetric. Sampling early leaves the FFT
    # window inside the prefix, which is a cyclic shift of the symbol
    # and so a pure phase ramp across subcarriers that the channel
    # estimator absorbs; sampling late drags the next symbol into the
    # window, which is intersymbol interference and nothing absorbs
    # that. Half of the clean prefix is the margin, and it is also
    # roughly what the channel's own delay spread wants
    tau = (tau - max(cw // 2, 4)) % slot
    cfo_frac = -np.angle(m[(tau + max(cw // 2, 4)) % slot]) \
        / (2 * np.pi) * (cfg.scs_khz * 1e3)
    return tau, cfo_frac


def fft_symbols(y, cfg, start, n_symbols):
    """Strip CPs and FFT n_symbols starting at sample `start` of a
    slot boundary. Returns (n_symbols, n_sc)."""
    cp0, cp = cp_lengths(cfg)
    n = cfg.n_fft
    half = cfg.n_sc // 2
    out = np.zeros((n_symbols, cfg.n_sc), dtype=complex)
    pos = start
    for l in range(n_symbols):
        cpl = cp0 if (l % 14) == 0 else cp
        pos += cpl
        seg = y[pos:pos + n]
        if seg.size < n:
            seg = np.pad(seg, (0, n - seg.size))
        spec = np.fft.fft(seg) / np.sqrt(n)
        out[l, half:] = spec[:half]
        out[l, :half] = spec[-half:]
        pos += n
    return out


def integer_cfo_and_slot(sym_grid, cfg, max_shift=4):
    """Integer-subcarrier CFO and absolute slot alignment from the
    DM-RS.

    The cyclic prefix says where a symbol starts but not which symbol
    it is, and its phase only resolves the carrier offset inside one
    subcarrier spacing. The DM-RS settles both, because its Gold
    sequence is seeded with the slot and symbol number: finding which
    reference matches tells the receiver not just that it is looking at
    a pilot but at *which* pilot, and the frequency shift that makes it
    match is the integer part of the offset.

    The metric is differential across neighbouring pilots. With
    v_k = r_k ref_k*, which is the channel times whatever phase ramp
    the timing backoff put there, the product v_k v_{k+1}* has a phase
    that does not depend on the ramp at all, so the sum stays coherent
    where a plain dot product would wind itself to zero over 306
    subcarriers.

    Returns (integer CFO in subcarriers, sample offset from the first
    symbol of the probe back to the start of slot 0), so the caller can
    align on absolute slot 0 even when the prefix fold locked onto a
    different symbol of a different slot."""
    offs = symbol_offsets(cfg)
    slot = slot_samples(cfg)
    refs = [(s, l, dmrs_sequence(cfg, s, l))
            for s in (0, 1) for l in cfg.dmrs_symbols]
    best = (0, 0, -1.0)
    for l in range(sym_grid.shape[0]):
        pos = (l // 14) * slot + offs[l % 14]
        for shift in range(-max_shift, max_shift + 1):
            rx = np.roll(sym_grid[l], -shift)[0::2]
            for s, ls, ref in refs:
                v = rx[:ref.size] * np.conj(ref)
                c = np.abs(np.sum(v[:-1] * np.conj(v[1:]))) \
                    / (np.sum(np.abs(v) ** 2) + 1e-12)
                if c > best[2]:
                    best = (shift, pos - (s * slot + offs[ls]), c)
    return best[0], best[1]


def estimate_channel(sym_grid, cfg, slot0):
    """LS at the DM-RS combs, DFT denoising across frequency, linear
    interpolation across time. The true channel lives inside the CP,
    so LS content beyond that delay is noise and is zeroed."""
    n_symbols = sym_grid.shape[0]
    cp0, cp = cp_lengths(cfg)
    h_at = {}
    for l in range(n_symbols):
        if (l % 14) in cfg.dmrs_symbols:
            slot = slot0 + l // 14
            ref = dmrs_sequence(cfg, slot, l % 14)
            ls = sym_grid[l, 0::2] / ref
            if cfg.chanest == "dft":
                # the timing backoff puts a linear phase ramp on H;
                # derotate it first so the delay-domain impulse sits
                # near zero, window with tapered edges, re-apply. A
                # brickwall on the raw ramped estimate rings at the
                # band edges and was measured to cost 4x in EVM.
                k = np.arange(ls.size)
                slope = np.angle(np.sum(ls[1:] * np.conj(ls[:-1])))
                flat = ls * np.exp(-1j * slope * k)
                imp = np.fft.ifft(flat)
                keep = max(int(np.ceil(cp * ls.size * 2 / cfg.n_fft)), 8)
                guard = max(ls.size // 48, 4)
                win = np.zeros(ls.size)
                win[:keep] = 1.0
                win[-guard:] = 1.0
                t = min(4, keep // 2)
                ramp = 0.5 * (1 + np.cos(np.pi * (np.arange(t) + 1) / (t + 1)))
                win[keep - t:keep] = ramp
                ls = np.fft.fft(imp * win) * np.exp(1j * slope * k)
            full = np.empty(cfg.n_sc, dtype=complex)
            full[0::2] = ls
            full[1::2] = 0.5 * (ls + np.roll(ls, -1))
            full[-1] = ls[-1]
            h_at[l] = full
    keys = sorted(h_at)
    h = np.zeros((n_symbols, cfg.n_sc), dtype=complex)
    for l in range(n_symbols):
        if l <= keys[0]:
            h[l] = h_at[keys[0]]
        elif l >= keys[-1]:
            h[l] = h_at[keys[-1]]
        else:
            k0 = max(k for k in keys if k <= l)
            k1 = min(k for k in keys if k >= l)
            w = 0.0 if k1 == k0 else (l - k0) / (k1 - k0)
            h[l] = (1 - w) * h_at[k0] + w * h_at[k1]
    return h


def equalize_and_demap(sym_grid, h, cfg, sigma2_re, slot0):
    """MMSE one-tap equalizer, optional decision-directed common
    phase per symbol, then per-RE max-log LLRs in transmit order."""
    from .mod import pam_levels, qam_map
    data_mask, _ = data_re_map(cfg, sym_grid.shape[0])
    h2 = np.abs(h) ** 2
    eq = np.conj(h) / (h2 + sigma2_re)
    z = sym_grid * eq
    scale = h2 / (h2 + sigma2_re)                  # MMSE bias
    z = z / np.maximum(scale, 1e-9)
    if cfg.cpe_track:
        lv = pam_levels(int(np.log2(cfg.qam)) // 2) \
            / np.sqrt({4: 2, 16: 10, 64: 42, 256: 170}[cfg.qam])
        for l in range(z.shape[0]):
            zs = z[l][data_mask[l]]
            di = lv[np.argmin(np.abs(zs.real[:, None] - lv), axis=1)]
            dq = lv[np.argmin(np.abs(zs.imag[:, None] - lv), axis=1)]
            d = di + 1j * dq
            rot = np.vdot(d, zs)
            z[l] *= np.exp(-1j * np.angle(rot))
    nvar_re = sigma2_re / np.maximum(h2, 1e-9)
    llrs = []
    for l in range(z.shape[0]):
        zl = z[l][data_mask[l]]
        nl = nvar_re[l][data_mask[l]]
        llrs.append(qam_llrs(zl[None], cfg.qam, nl[None])[0])
    return z, np.concatenate(llrs) if llrs else np.zeros(0)

"""Transmit chain after the grid: IFFT, cyclic prefix, WOLA
windowing, DAC with oversampling, the I/Q modulator's imperfections,
and a Rapp power amplifier.

Windowing is the quiet hero of OFDM spectra: a plain cyclic prefix
changes the waveform abruptly at every symbol boundary and the
spectrum pays for it in sidelobes; tapering W samples of each edge
with a raised cosine and overlap-adding neighbouring symbols buys
tens of dB of out-of-band rejection for no receiver change at all.
The PA is the Rapp model of a solid-state amplifier, driven at a
configured input backoff, which is where OFDM's peak-to-average
ratio turns into EVM and spectral regrowth.
"""

import numpy as np

from .grid import cp_lengths


def ofdm_modulate(grids, cfg):
    """grids (n_symbols, n_sc) frequency-domain -> serial waveform at
    the base rate, with CP and optional WOLA windowing."""
    n_symbols = grids.shape[0]
    cp0, cp = cp_lengths(cfg)
    n = cfg.n_fft
    spec = np.zeros((n_symbols, n), dtype=complex)
    half = cfg.n_sc // 2
    spec[:, :half] = grids[:, half:]               # DC and positive bins
    spec[:, -half:] = grids[:, :half]              # negative bins
    x = np.fft.ifft(spec, axis=1) * np.sqrt(n)
    w = cfg.window_samples
    total = sum((cp0 if (l % 14) == 0 else cp) + n for l in range(n_symbols))
    out = np.zeros(total + w, dtype=complex)
    taper = 0.5 * (1 - np.cos(np.pi * (np.arange(w) + 0.5) / w)) if w else None
    pos = 0
    for l in range(n_symbols):
        cpl = cp0 if (l % 14) == 0 else cp
        sym = np.concatenate([x[l, -cpl:], x[l]])
        if w:
            # WOLA: taper the head of the CP itself and grow a tapered
            # cyclic suffix that overlaps the next symbol's tapered CP,
            # so the symbol period is preserved and the window spends
            # CP margin, not time
            sym[:w] *= taper
            ext = np.concatenate([sym, x[l, :w] * taper[::-1]])
            out[pos:pos + ext.size] += ext
        else:
            out[pos:pos + sym.size] += sym
        pos += cpl + n
    return out[: total + w] if w else out[:total]


def dac_and_frontend(x, cfg, rng):
    """Oversample, quantise, apply I/Q imbalance with LO leakage, and
    push through the Rapp PA. Returns the waveform at os * fs."""
    n = x.size
    spec = np.fft.fft(x)
    up = np.zeros(n * cfg.os, dtype=complex)
    up[:n // 2] = spec[:n // 2]
    up[-n // 2:] = spec[-n // 2:]
    y = np.fft.ifft(up) * cfg.os
    if cfg.dac_bits and cfg.include_impairments:
        y = _quantise(y, cfg.dac_bits)
    if cfg.include_impairments:
        g = 10 ** (cfg.tx_iq_amp_db / 20)
        phi = np.deg2rad(cfg.tx_iq_phase_deg)
        y = y.real + 1j * g * (y.imag * np.cos(phi) + y.real * np.sin(phi))
        leak = 10 ** (cfg.tx_lo_leak_db / 20) * np.sqrt(np.mean(np.abs(y) ** 2))
        y = y + leak
    papr_db = 10 * np.log10(np.max(np.abs(y) ** 2)
                            / np.mean(np.abs(y) ** 2))
    if cfg.pa_p > 0 and cfg.include_impairments:
        rms = np.sqrt(np.mean(np.abs(y) ** 2))
        a_sat = rms * 10 ** (cfg.pa_ibo_db / 20)
        r = np.abs(y) / a_sat
        y = y / (1 + r ** (2 * cfg.pa_p)) ** (1 / (2 * cfg.pa_p))
    return y, papr_db


def _quantise(y, bits):
    def q(u):
        lo, hi = np.percentile(u, 0.02), np.percentile(u, 99.98)
        span = max(hi - lo, 1e-12) * 1.05
        step = span / (2 ** bits - 1)
        return np.clip(np.round((u - lo) / step), 0, 2 ** bits - 1) * step + lo
    return q(y.real) + 1j * q(y.imag)

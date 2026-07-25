"""The channel the block diagram writes as h(tau, t) + AWGN: a tap
delay line whose taps fade in time, plus white noise and a carrier
frequency offset.

Profiles are the TR 38.901 TDL tables (A/B/C are NLOS Rayleigh; D/E
carry a Rician first tap), with every normalised delay scaled by the
configured RMS delay spread. Each tap fades with a Jakes Doppler
spectrum built from a sum of sinusoids, so h varies within a slot
when the Doppler says it should; fractional tap delays land on the
sample grid through windowed-sinc interpolation so the profile is
honoured even when it does not divide the sample period.
"""

import numpy as np

from .tdl_tables import TDL

_K_LOS_DB = {"d": 13.3, "e": 22.0}   # TR 38.901 first-tap K-factors


def apply_channel(x, cfg, fs, rng):
    """x at rate fs -> (faded waveform, per-sample noise sigma^2 that
    gives cfg.snr_db per occupied subcarrier). The caller adds the
    noise so receive filtering stays outside this function."""
    if cfg.channel == "awgn" or not cfg.include_impairments:
        y = x.copy()
    else:
        key = cfg.channel.split("-")[1]
        prof = TDL[key]
        delays = np.asarray(prof["delays"]) * cfg.delay_spread_ns * 1e-9
        p_lin = 10 ** (np.asarray(prof["powers_db"]) / 10)
        p_lin = p_lin / p_lin.sum()
        k_db = _K_LOS_DB.get(key) if prof["los"] else None
        n = x.size
        t = np.arange(n) / fs
        y = np.zeros(n, dtype=complex)
        for i, (tau, p) in enumerate(zip(delays, p_lin)):
            if cfg.doppler_hz > 0:
                g = _jakes(cfg.doppler_hz, t, rng)
            else:
                g = (rng.standard_normal() + 1j * rng.standard_normal()) \
                    / np.sqrt(2.0)
            if i == 0 and k_db is not None:
                k = 10 ** (k_db / 10)
                g = np.sqrt(k / (k + 1)) + np.sqrt(1.0 / (k + 1)) * g
            gx = np.sqrt(p) * (g * x)
            d = tau * fs
            di, df = int(np.floor(d)), float(d - np.floor(d))
            span = 12
            ks = np.arange(-span, span + 1)
            h = np.sinc(ks - df) * np.hamming(ks.size)
            for kk, hh in zip(ks, h):
                if abs(hh) < 1e-4:
                    continue
                sh = di + int(kk)
                if sh >= 0:
                    y[sh:] += hh * gx[:n - sh]
                else:
                    y[:n + sh] += hh * gx[-sh:]
    if cfg.include_impairments and cfg.cfo_hz:
        y = y * np.exp(2j * np.pi * cfg.cfo_hz * np.arange(y.size) / fs)
    p_sig = float(np.mean(np.abs(y) ** 2))
    es = p_sig * (fs / (cfg.n_sc * cfg.scs_khz * 1e3))
    sigma2 = es / 10 ** (cfg.snr_db / 10)
    return y, sigma2


def _jakes(fd, t, rng, n_sin=16):
    """Sum-of-sinusoids Rayleigh fading with a Jakes spectrum."""
    alpha = rng.uniform(0, 2 * np.pi, n_sin)
    phi = rng.uniform(0, 2 * np.pi, n_sin)
    psi = rng.uniform(0, 2 * np.pi, n_sin)
    w = 2 * np.pi * fd * np.cos(alpha)
    re = np.cos(np.outer(t, w) + phi).sum(axis=1)
    im = np.sin(np.outer(t, w) + psi).sum(axis=1)
    return (re + 1j * im) / np.sqrt(n_sin)

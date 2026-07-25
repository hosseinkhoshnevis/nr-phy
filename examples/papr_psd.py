#!/usr/bin/env python3
"""The waveform's two classics: the PAPR CCDF and the transmit
spectrum with and without WOLA windowing and the PA.

    python examples/papr_psd.py
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from nrphy import SimConfig
from nrphy import phy_tx
from nrphy.mod import qam_map


def make_wave(cfg, rng, n_symbols=56):
    grid = qam_map(rng.integers(0, 2, (1, n_symbols * cfg.n_sc * 4),
                                dtype=np.uint8), 16).reshape(n_symbols,
                                                             cfg.n_sc)
    return phy_tx.ofdm_modulate(grid, cfg)


def psd(x, n=4096):
    segs = x[: (x.size // n) * n].reshape(-1, n)
    w = np.hanning(n)
    p = (np.abs(np.fft.fft(segs * w, axis=1)) ** 2).mean(axis=0)
    return np.fft.fftshift(np.fft.fftfreq(n)), np.fft.fftshift(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--papr-symbols", type=int, default=40000,
                    help="the CCDF cannot resolve below 1/this, and the "
                         "interesting part of the tail is at 1e-4")
    a = ap.parse_args()
    out_dir = Path(a.out_dir or Path(__file__).parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)

    # PAPR CCDF of the base OFDM waveform, one point per symbol. The
    # tail is the whole point of this curve, so it needs enough symbols
    # to resolve 1e-4, which is tens of thousands of them. That is more
    # waveform than is worth holding at once, so it goes in chunks: each
    # contributes its own peaks and its own mean power, and the ratio is
    # formed at the end against the pooled mean.
    cfg = SimConfig(include_impairments=False)
    period = 72 + 1024
    ths = np.arange(4.0, 14.0, 0.25)
    peaks, tot_pow, tot_n = [], 0.0, 0
    left, chunk = a.papr_symbols, 500
    while left > 0:
        n = min(chunk, left)
        x = make_wave(cfg, rng, n)
        segs = x[: (x.size // period) * period].reshape(-1, period)
        peaks.append((np.abs(segs) ** 2).max(axis=1))
        tot_pow += float((np.abs(x) ** 2).sum())
        tot_n += x.size
        left -= n
    papr = 10 * np.log10(np.concatenate(peaks) / (tot_pow / tot_n))
    ccdf = [(papr > t).mean() for t in ths]
    with open(out_dir / "papr_ccdf.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["papr_db", "ccdf"])
        for t, c in zip(ths, ccdf):
            w.writerow([t, c])

    # spectra: plain CP vs WOLA, and WOLA through the PA at 8 dB IBO.
    # Fresh generator so the spectra do not move when the PAPR run
    # above is asked for a different number of symbols.
    rng = np.random.default_rng(2)
    rows = {}
    for name, cfgv in (
        ("plain", SimConfig(window_samples=0, include_impairments=False)),
        ("wola", SimConfig(window_samples=18, include_impairments=False)),
        ("wola_pa", SimConfig(window_samples=18, pa_ibo_db=8.0)),
    ):
        xv = make_wave(cfgv, rng, 56)
        yv, _ = phy_tx.dac_and_frontend(xv, cfgv, rng)
        f, p = psd(yv)
        rows[name] = p / p.max()
        rows["freq"] = f * cfgv.os          # in units of base fs
    with open(out_dir / "psd.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["freq_fs", "plain", "wola", "wola_pa"])
        for i in range(rows["freq"].size):
            w.writerow([rows["freq"][i], rows["plain"][i],
                        rows["wola"][i], rows["wola_pa"][i]])
    print(f"wrote {out_dir}/papr_ccdf.csv and psd.csv")


if __name__ == "__main__":
    main()

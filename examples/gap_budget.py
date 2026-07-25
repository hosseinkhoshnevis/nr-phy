#!/usr/bin/env python3
"""Where the decibels go.

A coded cliff on its own says little: is the gap to capacity the
code, the decoder, the estimators, or the front ends? This script
takes the same transport block through four increasingly honest
settings and reports the Es/N0 each one needs for a 10% block error
rate, the operating point 3GPP link-level curves are drawn at.

    capacity   BICM mutual information of the constellation
    code       LDPC + rate matching + QAM + AWGN, exact LLRs, no OFDM
    ofdm       the full waveform and the blind receiver, ideal analog
    front end  the same run with DAC, I/Q, PA and CFO switched on

    python examples/gap_budget.py --qam 16 --rate 0.5
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from nrphy import SimConfig, run_link
from nrphy.fec import crc, ldpc
from nrphy.mod import bicm_capacity, qam_llrs, qam_map

TARGET = 0.1          # 10% BLER, the 3GPP link-level operating point


def capacity_threshold(qam, rate):
    """Es/N0 at which the BICM mutual information of this
    constellation equals the bits per symbol the run carries."""
    want = rate * np.log2(qam)
    lo, hi = -12.0, 40.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if bicm_capacity(qam, mid) < want:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def code_only_bler(qam, rate, snr_db, n_tb, seed):
    """The code and nothing else: encode, rate match, map, add AWGN
    at this Es/N0, demap with the true noise variance, decode."""
    rng = np.random.default_rng(seed)
    qm = int(np.log2(qam))
    e_bits = int(np.ceil(ldpc.K_BITS / rate / qm)) * qm
    a_bits = ldpc.K_BITS - 24
    payload = rng.integers(0, 2, (n_tb, a_bits), dtype=np.uint8)
    tb = np.concatenate([payload, crc.crc24a(payload)], axis=1)
    coded = ldpc.rate_match(ldpc.encode(tb), e_bits, qm)
    sym = qam_map(coded, qam)
    n0 = 10 ** (-snr_db / 10)                      # unit symbol energy
    noise = (rng.standard_normal(sym.shape)
             + 1j * rng.standard_normal(sym.shape)) * np.sqrt(n0 / 2)
    llr = qam_llrs(sym + noise, qam, n0).reshape(n_tb, e_bits)
    info, _ = ldpc.Decoder(e_bits, iters=40).decode(
        ldpc.rate_recover(llr, e_bits, qm))
    got = info[:, :a_bits]
    ok = (crc.crc24a(got) == info[:, a_bits:]).all(axis=1)
    return float((~ok).mean())


def chain_bler(qam, rate, snr_db, n_tb, seed, impair, channel="awgn"):
    r = run_link(SimConfig(qam=qam, code_rate=rate, snr_db=snr_db,
                           n_tb=n_tb, channel=channel,
                           include_impairments=impair,
                           cfo_hz=41000.0 if impair else 0.0), seed=seed)
    return r["post_fec"]["block_errors"] / r["post_fec"]["blocks"]


def threshold(fn, lo, hi, n_tb, tol=0.05):
    """Bisect on Es/N0 for the target BLER, averaging a few seeds."""
    for _ in range(7):
        mid = 0.5 * (lo + hi)
        b = np.mean([fn(mid, n_tb, s) for s in (1, 2, 3)])
        print(f"    {mid:6.2f} dB -> BLER {b:.3f}")
        if b > TARGET:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qam", type=int, default=16)
    ap.add_argument("--rate", type=float, default=0.5)
    ap.add_argument("--ntb", type=int, default=10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    q, r = a.qam, a.rate

    cap = capacity_threshold(q, r)
    print(f"{q}QAM rate {r}: BICM capacity threshold {cap:6.2f} dB")
    print("  code only")
    code = threshold(lambda s, n, sd: code_only_bler(q, r, s, n, sd),
                     cap - 1.0, cap + 6.0, a.ntb)
    print("  ofdm, ideal analog")
    ofdm = threshold(lambda s, n, sd: chain_bler(q, r, s, n, sd, False),
                     cap - 1.0, cap + 8.0, a.ntb)
    print("  ofdm, front ends and CFO on")
    full = threshold(lambda s, n, sd: chain_bler(q, r, s, n, sd, True),
                     cap - 1.0, cap + 8.0, a.ntb)
    rows = [{"stage": "bicm capacity", "esn0_db": round(cap, 2),
             "gap_db": 0.0},
            {"stage": "code only", "esn0_db": round(code, 2),
             "gap_db": round(code - cap, 2)},
            {"stage": "ofdm ideal analog", "esn0_db": round(ofdm, 2),
             "gap_db": round(ofdm - cap, 2)},
            {"stage": "full chain", "esn0_db": round(full, 2),
             "gap_db": round(full - cap, 2)}]
    for row in rows:
        print(f"{row['stage']:>20s}  {row['esn0_db']:6.2f} dB"
              f"  (+{row['gap_db']:.2f})")
    out = Path(a.out or Path(__file__).parents[1]
               / "docs" / "figures" / f"gap_budget_{q}_{r}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""BER and BLER against SNR, adaptive: transition points collect
blocks until --min-block-errors or --max-tb.

    python examples/snr_sweep.py --qam 16 --rate 0.5 --channel awgn
    python examples/snr_sweep.py --channel tdl-c --start 4 --stop 24
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from nrphy import SimConfig, run_link
from nrphy.mod import exact_ber_qam_awgn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qam", type=int, default=16)
    ap.add_argument("--rate", type=float, default=0.5)
    ap.add_argument("--channel", default="awgn")
    ap.add_argument("--start", type=float, default=6.0)
    ap.add_argument("--stop", type=float, default=16.0)
    ap.add_argument("--points", type=int, default=11)
    ap.add_argument("--grid", default=None)
    ap.add_argument("--tb", type=int, default=8, help="blocks per run")
    ap.add_argument("--min-tb", type=int, default=16)
    ap.add_argument("--max-tb", type=int, default=400)
    ap.add_argument("--min-block-errors", type=int, default=12)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.grid:
        grid = [float(x) for x in a.grid.split(",")]
    else:
        grid = list(np.linspace(a.start, a.stop, a.points))
    rows = []
    for snr in grid:
        pre_e = pre_b = tb = blk_e = post_e = post_b = 0
        seed = 100
        while tb < a.max_tb:
            r = run_link(SimConfig(qam=a.qam, code_rate=a.rate,
                                   channel=a.channel, snr_db=snr,
                                   n_tb=a.tb), seed=seed)
            seed += 1
            pre_e += r["pre_fec_errors"]; pre_b += r["pre_fec_bits"]
            p = r["post_fec"]
            blk_e += p["block_errors"]; tb += p["blocks"]
            post_e += p["errors"]; post_b += p["bits"]
            if tb >= a.min_tb and blk_e >= a.min_block_errors:
                break
            if tb >= a.min_tb and blk_e == 0 and tb >= 4 * a.min_tb:
                break
        rows.append({
            "snr_db": round(snr, 3),
            "pre_ber": pre_e / max(pre_b, 1),
            "post_ber_errors": post_e, "post_ber_bits": post_b,
            "block_errors": blk_e, "blocks": tb,
            "theory_uncoded_ber": exact_ber_qam_awgn(a.qam, snr)
            if a.channel == "awgn" else "",
        })
        print(f"snr {snr:5.1f}: pre {rows[-1]['pre_ber']:.3e}  "
              f"bler {blk_e}/{tb}  bits {post_e}/{post_b}")
    out = Path(a.out or Path(__file__).with_name(
        f"snr_sweep_{a.channel}_{a.qam}.csv"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

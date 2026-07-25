#!/usr/bin/env python3
"""EVM against PA input backoff, held against the 38.101 transmit
limits per modulation order. This is where OFDM's PAPR becomes a
design number: back off too little and the Rapp PA eats the
constellation, back off too much and every watt is wasted.

The sweep runs at 45 dB Es/N0 on purpose. The question here is what
the analog chain does, so thermal noise is pushed far enough down
(0.56%) that the curve's floor is the I/Q, LO and quantisation floor
rather than the receiver's own noise. That is the same operating
point the impairment-by-impairment table in the paper uses, so the
two can be read against each other.

    python examples/evm_backoff.py
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from nrphy import SimConfig, run_link

LIMITS = {4: 17.5, 16: 12.5, 64: 8.0, 256: 3.5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ibos",
                    default="2,3,4,5,6,7,8,9,10,11,12,14,16,18,20,24")
    ap.add_argument("--snr", type=float, default=45.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = []
    for ibo in [float(x) for x in a.ibos.split(",")]:
        row = {"ibo_db": ibo}
        for qam in (16, 64, 256):
            r = run_link(SimConfig(n_tb=1, qam=qam, snr_db=a.snr,
                                   pa_ibo_db=ibo, cfo_hz=0.0), seed=3)
            row[f"evm_{qam}"] = round(r["evm_percent"], 3)
        rows.append(row)
        print(f"IBO {ibo:4.1f} dB: " + "  ".join(
            f"{q}QAM {row[f'evm_{q}']:5.2f}%" for q in (16, 64, 256)))
    out = Path(a.out or Path(__file__).with_name("evm_backoff.csv"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out}  (38.101 limits: "
          + ", ".join(f"{q}QAM {v}%" for q, v in LIMITS.items()) + ")")


if __name__ == "__main__":
    main()

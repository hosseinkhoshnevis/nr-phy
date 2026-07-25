#!/usr/bin/env python3
"""One link, all numbers.

    python examples/run_single.py
    python examples/run_single.py --qam 64 --rate 0.75 --snr 22 --channel tdl-c
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from nrphy import SimConfig, run_link


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qam", type=int, default=16)
    ap.add_argument("--rate", type=float, default=0.5)
    ap.add_argument("--snr", type=float, default=12.0)
    ap.add_argument("--channel", default="awgn")
    ap.add_argument("--doppler", type=float, default=70.0)
    ap.add_argument("--cfo", type=float, default=41000.0)
    ap.add_argument("--tb", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    cfg = SimConfig(qam=a.qam, code_rate=a.rate, snr_db=a.snr,
                    channel=a.channel, doppler_hz=a.doppler,
                    cfo_hz=a.cfo, n_tb=a.tb)
    r = run_link(cfg, seed=a.seed)
    print(f"{a.qam}-QAM rate {a.rate} over {a.channel} at {a.snr} dB, "
          f"{r['n_slots']} slots")
    print(f"sync: timing {r['timing_sample']} (true {r['true_delay']}), "
          f"CFO error {r['cfo_err_hz']:+.0f} Hz of {cfg.cfo_hz:.0f} applied")
    print(f"PAPR {r['papr_db']:.1f} dB   EVM {r['evm_percent']:.1f}%")
    print(f"pre-FEC BER {r['pre_fec_ber']:.3e} "
          f"({r['pre_fec_errors']}/{r['pre_fec_bits']})")
    p = r["post_fec"]
    print(f"post-FEC: {p['errors']}/{p['bits']} bits, "
          f"BLER {p['block_errors']}/{p['blocks']} by CRC24A")


if __name__ == "__main__":
    main()

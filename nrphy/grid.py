"""The resource grid of TS 38.211: numerology, cyclic prefixes,
type-1 DM-RS on its Gold sequence, and the mapping of coded symbols
onto the grid.

A slot is 14 OFDM symbols; at 30 kHz subcarrier spacing the first
symbol of every slot carries the longer cyclic prefix so that the
slot lands on exactly 0.5 ms. DM-RS is config type 1: QPSK pilots on
the even subcarriers of the configured symbols, drawn from the
length-31 Gold sequence with the standard's c_init, so the pilots are
bit-exact 38.211 and a real UE would recognise them.
"""

import numpy as np


def cp_lengths(cfg):
    """(cp_first, cp_rest) in samples for one slot of 14 symbols."""
    mu = int(round(np.log2(cfg.scs_khz / 15.0)))
    scale = cfg.n_fft / 2048.0
    cp = int(round(144 * scale))
    cp0 = cp + int(round(16 * scale * (1 << mu)))
    return cp0, cp


def slot_samples(cfg):
    cp0, cp = cp_lengths(cfg)
    return (cp0 + cfg.n_fft) + 13 * (cp + cfg.n_fft)


def gold(c_init, n):
    """38.211 clause 5.2.1 length-31 Gold sequence, first n bits."""
    nc = 1600
    x1 = np.zeros(nc + n + 31, dtype=np.uint8)
    x2 = np.zeros(nc + n + 31, dtype=np.uint8)
    x1[0] = 1
    for i in range(31):
        x2[i] = (c_init >> i) & 1
    for i in range(nc + n):
        x1[i + 31] = x1[i + 3] ^ x1[i]
        x2[i + 31] = x2[i + 3] ^ x2[i + 2] ^ x2[i + 1] ^ x2[i]
    return x1[nc:nc + n] ^ x2[nc:nc + n]


def dmrs_sequence(cfg, slot, l):
    """Type-1 DM-RS QPSK values for the even subcarriers of symbol l,
    c_init per clause 7.4.1.1.1 (n_SCID = 0)."""
    c_init = ((1 << 17) * (14 * slot + l + 1) * (2 * cfg.n_id + 1)
              + 2 * cfg.n_id) % (1 << 31)
    n_pil = cfg.n_sc // 2
    c = gold(c_init, 2 * n_pil).astype(np.float64)
    return ((1 - 2 * c[0::2]) + 1j * (1 - 2 * c[1::2])) / np.sqrt(2.0)


def scrambling(cfg, tb_index, n):
    """Data scrambling sequence (clause 7.3.1.1 shape, RNTI stands in
    as the transport-block index so every block is distinct)."""
    c_init = ((tb_index + 1) * (1 << 15) + cfg.n_id) % (1 << 31)
    return gold(c_init, n)


def data_re_map(cfg, n_symbols):
    """Boolean masks (n_symbols, n_sc): which REs carry data and which
    carry DM-RS, for symbols counted from the start of a slot."""
    data = np.ones((n_symbols, cfg.n_sc), dtype=bool)
    pilot = np.zeros((n_symbols, cfg.n_sc), dtype=bool)
    for l in range(n_symbols):
        if (l % 14) in cfg.dmrs_symbols:
            pilot[l, 0::2] = True
            data[l, 0::2] = False
    return data, pilot


def data_res_per_slot(cfg):
    data, _ = data_re_map(cfg, 14)
    return int(data.sum())

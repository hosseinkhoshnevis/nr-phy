"""CRCs of TS 38.212: CRC24A attaches to the transport block and is
the receiver's block-error verdict, exactly as the standard uses it.
Registers initialise to zero, bits go MSB-first, batch-vectorised.
"""

import numpy as np

POLY24A = 0x1864CFB
POLY16 = 0x1021


def _crc(bits, poly, width):
    bits = np.asarray(bits, dtype=np.uint32)
    if bits.ndim == 1:
        bits = bits[None]
    reg = np.zeros(bits.shape[0], dtype=np.uint64)
    top = np.uint64(1 << (width - 1))
    mask = np.uint64((1 << width) - 1)
    p = np.uint64(poly)
    for k in range(bits.shape[1]):
        fb = ((reg & top) != 0) ^ (bits[:, k] != 0)
        reg = (reg << np.uint64(1)) & mask
        reg = np.where(fb, reg ^ p, reg)
    return reg


def crc24a(bits):
    """(B, n) or (n,) bits -> (B, 24) CRC bits, MSB first."""
    reg = _crc(bits, POLY24A, 24)
    return ((reg[:, None] >> np.arange(23, -1, -1).astype(np.uint64))
            & np.uint64(1)).astype(np.uint8)


def crc16(bits):
    reg = _crc(bits, POLY16, 16)
    return ((reg[:, None] >> np.arange(15, -1, -1).astype(np.uint64))
            & np.uint64(1)).astype(np.uint8)

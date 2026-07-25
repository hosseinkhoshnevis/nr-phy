"""LDPC, rate matching and CRC against the 38.212 structure."""

import numpy as np
import pytest

from nrphy.fec import crc, ldpc


def test_encoder_makes_valid_codewords():
    rng = np.random.default_rng(0)
    u = rng.integers(0, 2, (2, ldpc.K_BITS), dtype=np.uint8)
    assert ldpc.check(ldpc.encode(u)) == 0


def test_rate_match_interleaver_roundtrip():
    rng = np.random.default_rng(1)
    u = rng.integers(0, 2, (2, ldpc.K_BITS), dtype=np.uint8)
    cw = ldpc.encode(u)
    for qm, rate in ((2, 0.5), (4, 0.5), (6, 0.75), (8, 0.35)):
        e = int(np.ceil(ldpc.K_BITS / rate / qm)) * qm
        f = ldpc.rate_match(cw, e, qm)
        llr = 1.0 - 2.0 * f.astype(np.float32)
        buf = ldpc.rate_recover(llr, e, qm)
        d = cw[:, 2 * ldpc.Z:]
        sent = np.zeros(ldpc.N_CB, dtype=bool)
        sent[np.arange(e) % ldpc.N_CB] = True
        hard = (buf < 0).astype(np.uint8)
        assert (hard[:, sent] == d[:, sent]).all()
        assert (buf[:, ~sent] == 0).all()


def test_repetition_combines_llrs():
    rng = np.random.default_rng(2)
    u = rng.integers(0, 2, (1, ldpc.K_BITS), dtype=np.uint8)
    cw = ldpc.encode(u)
    e = int(np.ceil(1.5 * ldpc.N_CB / 4)) * 4     # wraps the buffer
    f = ldpc.rate_match(cw, e, 4)
    llr = 1.0 - 2.0 * f.astype(np.float32)
    buf = ldpc.rate_recover(llr, e, 4)
    wrapped = np.arange(e) % ldpc.N_CB
    twice = np.bincount(wrapped, minlength=ldpc.N_CB) == 2
    assert np.isclose(np.abs(buf[0, twice]), 2.0).all()


@pytest.mark.parametrize("rate,sigma", [(0.35, 0.95), (0.5, 0.72),
                                        (0.75, 0.5)])
def test_decode_across_rates(rate, sigma):
    rng = np.random.default_rng(int(rate * 100))
    u = rng.integers(0, 2, (2, ldpc.K_BITS), dtype=np.uint8)
    e = int(np.ceil(ldpc.K_BITS / rate / 4)) * 4
    f = ldpc.rate_match(ldpc.encode(u), e, 4)
    x = (1 - 2.0 * f.astype(np.float32)) \
        + sigma * rng.standard_normal(f.shape).astype(np.float32)
    raw = float(((x < 0) != (f > 0)).mean())
    buf = ldpc.rate_recover(2 * x / sigma ** 2, e, 4)
    info, ok = ldpc.Decoder(e).decode(buf)
    assert raw > 0.015                          # genuinely noisy input
    assert ok.all() and (info == u).all()


def test_crc24a_is_linear_and_detects_flips():
    rng = np.random.default_rng(3)
    a = rng.integers(0, 2, (1, 500), dtype=np.uint8)
    b = rng.integers(0, 2, (1, 500), dtype=np.uint8)
    assert (crc.crc24a(a ^ b) == crc.crc24a(a) ^ crc.crc24a(b)).all()
    ref = crc.crc24a(a)
    for k in (0, 250, 499):
        q = a.copy()
        q[0, k] ^= 1
        assert (crc.crc24a(q) != ref).any()

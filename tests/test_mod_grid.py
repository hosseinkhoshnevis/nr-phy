"""38.211 constellations, numerology, and the DM-RS machinery."""

import numpy as np

from nrphy import SimConfig
from nrphy.grid import (cp_lengths, dmrs_sequence, gold, slot_samples)
from nrphy.mod import exact_ber_qam_awgn, pam_levels, qam_map


def test_standard_constellation_points():
    # 38.211 16QAM: 0000 -> (1+j)/sqrt(10); 1010 -> (-3+j)/sqrt(10);
    # 1111 -> (-3-3j)/sqrt(10)
    s = qam_map(np.array([[0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1]],
                         dtype=np.uint8), 16)[0]
    assert np.isclose(s[0], (1 + 1j) / np.sqrt(10))
    assert np.isclose(s[1], (-3 + 1j) / np.sqrt(10))
    assert np.isclose(s[2], (-3 - 3j) / np.sqrt(10))
    # 64QAM bits 000000 -> (3+3j)/sqrt(42)
    s = qam_map(np.zeros((1, 6), dtype=np.uint8), 64)[0]
    assert np.isclose(s[0], (3 + 3j) / np.sqrt(42))
    # unit average power for every order
    rng = np.random.default_rng(0)
    for qam in (4, 16, 64, 256):
        qm = int(np.log2(qam))
        b = rng.integers(0, 2, (1, 6000 * qm // qm * qm), dtype=np.uint8)
        s = qam_map(b, qam)
        assert abs(np.mean(np.abs(s) ** 2) - 1.0) < 0.02


def test_gray_neighbours_differ_by_one_bit():
    for ax in (1, 2, 3, 4):
        lv = pam_levels(ax)
        order = np.argsort(lv)
        for a, b in zip(order[:-1], order[1:]):
            assert bin(int(a) ^ int(b)).count("1") == 1


def test_numerology_lands_on_half_a_millisecond():
    cfg = SimConfig()
    assert cp_lengths(cfg) == (88, 72)
    assert slot_samples(cfg) == 15360          # 0.5 ms at 30.72 Msps
    cfg2 = SimConfig(scs_khz=15.0, n_fft=2048)
    assert cp_lengths(cfg2) == (160, 144)


def test_gold_sequence_is_balanced_and_seeded():
    c = gold(12345, 20000)
    assert abs(c.mean() - 0.5) < 0.02
    assert (gold(12345, 100) == gold(12345, 100)).all()
    assert (gold(12345, 100) != gold(54321, 100)).any()


def test_dmrs_differs_per_symbol_and_slot():
    cfg = SimConfig()
    a = dmrs_sequence(cfg, 0, 2)
    b = dmrs_sequence(cfg, 0, 11)
    c = dmrs_sequence(cfg, 1, 2)
    assert a.size == cfg.n_sc // 2
    assert np.isclose(np.mean(np.abs(a) ** 2), 1.0)
    assert np.abs(np.vdot(a, b)) / a.size < 0.2
    assert np.abs(np.vdot(a, c)) / a.size < 0.2


def test_exact_qam_theory_sanity():
    # QPSK at Es/N0 = 9.6 dB (Eb/N0 6.6 dB) sits near 2.3e-3
    assert 1e-3 < exact_ber_qam_awgn(4, 9.6) < 4e-3
    assert exact_ber_qam_awgn(16, 12.0) > exact_ber_qam_awgn(4, 12.0)
    assert exact_ber_qam_awgn(64, 30.0) < 1e-9

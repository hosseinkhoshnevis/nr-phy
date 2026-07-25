"""End to end: theory anchor, sync, fading, and the front ends."""

import numpy as np
import pytest

from nrphy import SimConfig, run_link
from nrphy.mod import exact_ber_qam_awgn


def _clean(**kw):
    base = dict(n_tb=2, include_impairments=False)
    base.update(kw)
    return SimConfig(**base)


def test_chain_matches_qam_theory_on_awgn():
    # pre-FEC BER of the blind chain against the exact 16QAM curve;
    # the margin allows the ~0.3 dB of estimation the receiver pays
    cfg = _clean(snr_db=13.0)
    e = b = 0
    for s in (1, 2, 3):
        r = run_link(cfg, seed=s)
        e += r["pre_fec_errors"]
        b += r["pre_fec_bits"]
    th = exact_ber_qam_awgn(16, 13.0)
    assert e / b < 2.2 * th
    assert e / b > 0.5 * th


def test_evm_is_noise_limited_when_clean():
    r = run_link(_clean(snr_db=30.0), seed=1)
    assert 2.5 < r["evm_percent"] < 4.0        # 10^(-30/20) = 3.16%


def test_sync_finds_timing_and_cfo():
    for seed in (2, 3, 4):
        r = run_link(SimConfig(n_tb=2, snr_db=12), seed=seed)
        err = r["timing_sample"] - r["true_delay"]
        assert -60 <= err <= 0                 # early into the CP only
        assert abs(r["cfo_err_hz"]) < 500.0    # of 41 kHz applied
        assert r["post_fec"]["block_errors"] == 0


@pytest.mark.parametrize("qam,n_tb,slots", [(4, 8, 9), (16, 12, 7)])
def test_sync_does_not_drift_over_a_long_capture(qam, n_tb, slots):
    # the regression this guards: the prefix fold used to repeat at the
    # symbol period, but a slot is fourteen symbols plus the sixteen
    # extra samples of its long first prefix, so the fold slid by
    # sixteen samples every slot and the timing peak walked late in
    # proportion to the capture. Ask for enough transport blocks to
    # need nine slots and the error has to stay put, early inside the
    # prefix, with EVM still at the noise floor
    r = run_link(_clean(qam=qam, n_tb=n_tb, snr_db=30.0), seed=1)
    assert r["n_slots"] >= slots
    err = r["timing_sample"] - r["true_delay"]
    assert -60 <= err <= 0
    assert r["evm_percent"] < 4.5              # 3.16% is the floor
    assert r["post_fec"]["block_errors"] == 0


def test_sync_lands_on_absolute_slot_zero():
    # the second half of the same regression: the prefix fold can lock
    # onto any symbol of any slot, since thirteen of the fourteen fold
    # offsets still line up one symbol over. The DM-RS is what breaks
    # the tie, so it is searched against both slot parities, and a miss
    # shows up as a whole slot of timing error and a dead block
    for seed in (1, 2, 3, 4, 5):
        r = run_link(_clean(qam=4, n_tb=4, snr_db=30.0), seed=seed)
        err = r["timing_sample"] - r["true_delay"]
        assert abs(err) < 1000                 # a slot is 15360 samples
        assert r["post_fec"]["block_errors"] == 0


def test_evm_meets_the_38101_requirement_per_order():
    # TX EVM limits: QPSK 17.5%, 16QAM 12.5%, 64QAM 8%, 256QAM 3.5%;
    # at high SNR the chain's EVM is impairment-limited and must fit
    limits = {4: 17.5, 16: 12.5, 64: 8.0}
    for qam, lim in limits.items():
        r = run_link(SimConfig(n_tb=1, qam=qam, snr_db=35,
                               pa_ibo_db=9.0), seed=5)
        assert r["evm_percent"] < lim


def test_pa_backoff_trades_evm():
    hard = run_link(SimConfig(n_tb=1, snr_db=35, pa_ibo_db=3.0), seed=6)
    soft = run_link(SimConfig(n_tb=1, snr_db=35, pa_ibo_db=10.0), seed=6)
    assert hard["evm_percent"] > soft["evm_percent"]


def test_windowing_lowers_out_of_band_power():
    from nrphy import phy_tx
    rng = np.random.default_rng(0)
    grid = (rng.standard_normal((14, 612)) + 1j *
            rng.standard_normal((14, 612))) / np.sqrt(2)
    def oob(w):
        cfg = SimConfig(window_samples=w)
        x = phy_tx.ofdm_modulate(grid, cfg)
        f, p = _psd(x)
        band = np.abs(f) < 0.35
        far = np.abs(f) > 0.42
        return 10 * np.log10(p[far].mean() / p[band].mean())
    def _psd(x, n=2048):
        segs = x[: (x.size // n) * n].reshape(-1, n)
        w = np.hanning(n)
        return (np.fft.fftshift(np.fft.fftfreq(n)),
                np.fft.fftshift(np.abs(np.fft.fft(segs * w, axis=1)) ** 2)
                .mean(axis=0))
    assert oob(18) < oob(0) - 8.0              # windowing earns real dB


@pytest.mark.parametrize("chan", ["tdl-a", "tdl-c"])
def test_fading_channels_close_at_reasonable_snr(chan):
    r = run_link(SimConfig(n_tb=2, snr_db=17, channel=chan,
                           doppler_hz=70.0), seed=7)
    assert r["post_fec"]["block_errors"] == 0


def test_bler_orders_with_snr_on_fading():
    lo = run_link(SimConfig(n_tb=4, snr_db=6, channel="tdl-c"), seed=8)
    hi = run_link(SimConfig(n_tb=4, snr_db=20, channel="tdl-c"), seed=8)
    assert hi["post_fec"]["block_errors"] <= lo["post_fec"]["block_errors"]


def test_qam_orders_run_end_to_end():
    for qam, snr in ((4, 8), (64, 22), (256, 30)):
        r = run_link(SimConfig(n_tb=1, qam=qam, snr_db=snr,
                               code_rate=0.5), seed=9)
        assert r["post_fec"]["block_errors"] == 0

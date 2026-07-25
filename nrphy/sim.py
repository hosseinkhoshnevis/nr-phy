"""End to end, exactly the block diagram: bits, FEC encoding, the
rate-matching interleaver, QAM mapping, pilots onto the grid, IFFT,
cyclic prefix, windowing, DAC, I/Q modulator, PA, the fading channel
with AWGN and CFO, then the receiver earning its way back: LNA-side
noise, I/Q demodulation, AGC and ADC, CP-based sync, FFT, channel
estimation, the one-tap equalizer, per-RE LLRs, de-interleaving and
LDPC decoding, with CRC24A delivering the block verdict.

The transport block is one BG1 code block (A = 8424 bits + CRC24A);
E follows from the configured rate and fills as many data REs as it
needs, transport blocks back to back across slots. The receiver gets
no genie: timing and fractional CFO come from the cyclic prefix,
slot identity and integer CFO from the DM-RS, noise variance from
the pilot residuals. The known transmit symbols are used only to
measure EVM, never to receive.
"""

import numpy as np

from . import phy_rx, phy_tx
from .channel import apply_channel
from .config import SimConfig
from .fec import crc, ldpc
from .grid import (cp_lengths, data_re_map, data_res_per_slot,
                   dmrs_sequence, scrambling, slot_samples)
from .mod import qam_map


def run_link(cfg: SimConfig = None, seed=1):
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(seed)
    qm = int(np.log2(cfg.qam))

    # ---- transport: bits -> FEC -> rate match -> scramble -> QAM
    e_bits = int(np.ceil(ldpc.K_BITS / cfg.code_rate / qm)) * qm
    a_bits = ldpc.K_BITS - 24
    payload = rng.integers(0, 2, (cfg.n_tb, a_bits), dtype=np.uint8)
    tb = np.concatenate([payload, crc.crc24a(payload)], axis=1)
    coded = ldpc.rate_match(ldpc.encode(tb), e_bits, qm)
    for i in range(cfg.n_tb):
        coded[i] ^= scrambling(cfg, i, e_bits)
    syms = qam_map(coded, cfg.qam).reshape(-1)

    # ---- grid: data + pilots across as many slots as needed
    per_slot = data_res_per_slot(cfg)
    n_slots = int(np.ceil(syms.size / per_slot))
    n_symbols = 14 * n_slots
    data_mask, pilot_mask = data_re_map(cfg, n_symbols)
    grid = np.zeros((n_symbols, cfg.n_sc), dtype=complex)
    for l in range(n_symbols):
        if (l % 14) in cfg.dmrs_symbols:
            grid[l, 0::2] = dmrs_sequence(cfg, l // 14, l % 14)
    filler = qam_map(rng.integers(0, 2, (1, (data_mask.sum() - syms.size) * qm),
                                  dtype=np.uint8), cfg.qam).reshape(-1) \
        if data_mask.sum() > syms.size else np.zeros(0)
    grid[data_mask] = np.concatenate([syms, filler])

    # ---- waveform: IFFT + CP + window -> DAC -> I/Q -> PA
    x = phy_tx.ofdm_modulate(grid, cfg)
    delay = int(rng.integers(cp_lengths(cfg)[1], slot_samples(cfg) // 2))
    x = np.concatenate([np.zeros(delay, dtype=complex), x,
                        np.zeros(cfg.n_fft, dtype=complex)])
    tx_wave, papr_db = phy_tx.dac_and_frontend(x, cfg, rng)

    # ---- channel h(tau, t) + CFO, then the receive front end
    y, _ = apply_channel(tx_wave, cfg, cfg.fs * cfg.os, rng)
    y = phy_rx.rx_frontend(y, cfg, rng)

    # ---- sync: CP gives timing and fractional CFO; DM-RS the rest
    tau, cfo_frac = phy_rx.cp_sync(y, cfg, n_symbols)
    y1 = y * np.exp(-2j * np.pi * cfo_frac * np.arange(y.size) / cfg.fs)
    probe = phy_rx.fft_symbols(y1, cfg, tau, min(16, n_symbols + 2))
    k_int, slot_shift = phy_rx.integer_cfo_and_slot(probe, cfg)
    cfo_est = cfo_frac + k_int * cfg.scs_khz * 1e3
    y1 = y * np.exp(-2j * np.pi * cfo_est * np.arange(y.size) / cfg.fs)
    # the prefix fold gives a symbol boundary; the DM-RS says which
    # symbol of which slot it was, and slot_shift walks back to slot 0
    start = max(tau + slot_shift, 0)
    sym_grid = phy_rx.fft_symbols(y1, cfg, start, n_symbols)

    # ---- channel estimation and noise from the pilots themselves
    h = phy_rx.estimate_channel(sym_grid, cfg, 0)
    res, cnt = 0.0, 0
    for l in range(n_symbols):
        if (l % 14) in cfg.dmrs_symbols:
            ref = dmrs_sequence(cfg, l // 14, l % 14)
            res += float(np.sum(np.abs(sym_grid[l, 0::2]
                                       - h[l, 0::2] * ref) ** 2))
            cnt += ref.size
    sigma2_re = max(res / max(cnt, 1), 1e-12)

    # ---- equalize, demap, descramble, rate-recover, decode
    z, llr_stream = phy_rx.equalize_and_demap(sym_grid, h, cfg,
                                              sigma2_re, 0)
    llr_tb = llr_stream[: cfg.n_tb * e_bits].reshape(cfg.n_tb, e_bits)
    for i in range(cfg.n_tb):
        llr_tb[i] = llr_tb[i] * (1.0 - 2.0 * scrambling(cfg, i, e_bits))
    pre_hard = (llr_tb < 0).astype(np.uint8)
    pre_err = int((pre_hard != coded ^ np.array(
        [scrambling(cfg, i, e_bits) for i in range(cfg.n_tb)])).sum())
    buf = ldpc.rate_recover(llr_tb, e_bits, qm)
    dec = ldpc.Decoder(e_bits, iters=cfg.ldpc_iters)
    info, ok = dec.decode(buf)
    got_payload = info[:, :a_bits]
    crc_ok = (crc.crc24a(got_payload) == info[:, a_bits:]).all(axis=1)

    # ---- measurement (the reference symbols are used only here)
    zs = z[data_mask][: syms.size]
    evm = float(np.sqrt(np.mean(np.abs(zs - syms) ** 2)
                        / np.mean(np.abs(syms) ** 2)))
    post_err = int((got_payload != payload).sum())
    return {
        "pre_fec_ber": pre_err / coded.size,
        "pre_fec_errors": pre_err, "pre_fec_bits": int(coded.size),
        "post_fec": {
            "errors": post_err, "bits": int(payload.size),
            "ber": post_err / payload.size,
            "block_errors": int((~crc_ok).sum()), "blocks": cfg.n_tb,
            "decoder_converged": int(ok.sum()),
        },
        "evm_rms": evm, "evm_percent": 100 * evm,
        "papr_db": papr_db,
        "cfo_est_hz": float(cfo_est), "cfo_err_hz": float(cfo_est - cfg.cfo_hz)
        if cfg.include_impairments else 0.0,
        "timing_sample": int(start), "true_delay": delay,
        "n_slots": n_slots,
        "snr_db": cfg.snr_db,
    }

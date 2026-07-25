# nrphy

**Hossein Khoshnevis** · MIT licence

Baseband PHY simulation of a 5G NR OFDM transceiver in plain
vectorised numpy: the block diagram everyone draws on the whiteboard,
implemented to the 3GPP letter where the standard defines bits.
Third of a family: `ofm-phy` simulates the 800G coherent fibre
interface and `fso-phy` the SDA-OCT free-space link; this one
brings the same discipline to the
cellular air interface, and its LDPC is literally the same code the
free-space package borrowed from 5G, now back home.

## The chain

TX: bits -> LDPC encode (TS 38.212 BG1) -> rate-match + interleave ->
scramble -> QAM map (TS 38.211) -> resource grid + DM-RS pilots ->
IFFT -> cyclic prefix -> WOLA window -> DAC -> I/Q modulator -> PA
(Rapp) -> channel h(tau, t) + AWGN + CFO -> LNA/AGC -> I/Q demod ->
ADC -> CP-based sync (timing + fractional CFO) -> FFT -> DM-RS
integer CFO + slot identity -> channel estimation (LS + DFT denoise)
-> one-tap MMSE EQ -> per-RE max-log LLRs -> descramble ->
rate-recover -> LDPC decode -> CRC24A block verdict.

## What is inside

* **Bit-exact 38.212 coding**: the NR LDPC (base graph 1, Z = 384,
  K = 8448), CRC24A on the transport block, the circular-buffer rate
  matching with repetition for low rates, and the Q_m-row bit
  interleaver that spreads code bits over constellation levels.
  Verified: encoder syndromes, interleaver roundtrip, repetition LLR
  combining, decoding across rates.
* **Bit-exact 38.211 modulation and pilots**: the QAM constellations
  from the standard's per-axis recursion (unit tests pin the exact
  points), type-1 comb-2 DM-RS drawn from the length-31 Gold sequence
  with the standard's c_init per slot and symbol, normal CP with the
  long first-of-slot prefix so a slot is exactly 0.5 ms, and Gold
  data scrambling.
* **The front ends the diagram promises**: WOLA windowing that spends
  CP margin to buy tens of dB of out-of-band rejection, DAC
  oversampling and quantisation, I/Q modulator gain/phase imbalance
  with LO leakage, a Rapp solid-state PA at configured backoff, and
  the receive-side mirror (I/Q demod imbalance, AGC, ADC).
* **h(tau, t) + AWGN, as written**: TR 38.901 TDL-A/B/C/D/E profiles
  (delays scaled by the configured delay spread, D/E with their
  Rician first tap), per-tap Jakes Doppler fading, fractional-delay
  taps via windowed sinc, CFO, and noise calibrated so the configured
  Es/N0 per occupied subcarrier is exact by construction.
* **A receiver that earns its way back, blind**: van de Beek CP
  correlation for timing and fractional CFO, folded at the slot rather
  than the symbol (the long first prefix means a symbol-period fold
  slides 16 samples every slot and walks the peak late in proportion
  to the capture) and correlated only over the untapered part of each
  prefix so WOLA cannot bias it; absolute slot identity and integer
  CFO from the DM-RS, searched against both slot parities and both
  pilot symbols through a differential metric that is immune to the
  timing phase ramp; LS channel estimation with a
  ramp-aware DFT denoiser (the naive brickwall was measured to cost
  4x in EVM and stays out); linear time interpolation between DM-RS
  symbols; unbiased MMSE one-tap equalization; decision-directed
  common-phase tracking; per-RE noise variance from the pilot
  residuals. The known transmit symbols are used for EVM measurement
  only, never for reception.
* **Measurement discipline**: pre-FEC BER on coded bits, post-FEC BER
  and BLER by CRC24A (the standard's own verdict), EVM per 38.101
  held against the standard's per-order limits, PAPR CCDF, and an
  exact uncoded Gray-QAM theory curve computed from decision regions
  (not an approximation) as the AWGN anchor.

Everything is numpy. A run of 8 transport blocks (67k payload bits,
5 slots of 20 MHz at 30 kHz SCS) takes about a second on a laptop
core; TDL fading adds the tap convolutions.

## Install

```
pip install -e .          # numpy only
pip install -e .[dev]     # + matplotlib and pytest
```

## Quick start

```python
from nrphy import SimConfig, run_link

r = run_link(SimConfig())                      # 16QAM r=1/2, AWGN 12 dB
print(r["post_fec"]["block_errors"], r["evm_percent"])

r = run_link(SimConfig(channel="tdl-c", snr_db=16, doppler_hz=70))
```

Command line:

```
python examples/run_single.py --qam 64 --rate 0.75 --snr 22 --channel tdl-c
python examples/snr_sweep.py --qam 16 --rate 0.5 --channel awgn
python examples/papr_psd.py
python examples/evm_backoff.py
python examples/gap_budget.py --qam 16 --rate 0.5
```

`gap_budget.py` is the one worth running first if you want to know
whether the package is honest. It takes the same transport block
through four settings and bisects the Es/N0 each needs for 10% BLER:
the BICM capacity of the constellation, the code on its own with
exact LLRs and no OFDM, the full waveform with the blind receiver
and ideal analog, and the same run with the DAC, I/Q, PA and CFO
switched on. The rungs say where the decibels actually go, which is
more useful than a single cliff number.

## Layout

```
nrphy/
  config.py        SimConfig, every knob, 20 MHz / 30 kHz defaults
  sim.py           run_link: bits -> slots -> vacuum of the air -> BER
  fec/
    bg1.py         the 3GPP BG1 exponent table (Z=384)
    ldpc.py        encoder, NR rate matching + interleaver, min-sum
    crc.py         CRC24A (transport block), CRC16
  grid.py          numerology, CP lengths, Gold sequences, DM-RS, maps
  mod.py           38.211 QAM, per-RE LLRs, exact theory BER, BICM MI
  phy_tx.py        IFFT/CP/WOLA, DAC, I/Q modulator, Rapp PA
  channel.py       TR 38.901 TDL profiles, Jakes fading, CFO, AWGN
  phy_rx.py        front end, CP sync, FFT, chanest, MMSE EQ, CPE
  tdl_tables.py    the 38.901 delay/power tables
examples/          run_single, snr_sweep, papr_psd, evm_backoff,
                   gap_budget (capacity -> code -> OFDM -> front ends)
tests/             26 tests: standard structures, exact constellation
                   points, numerology, theory match, sync, fading
docs/              measured data (CSV) and the paper (docs/paper/)
```

## Simulation conventions worth knowing

One transport block is one BG1 code block (A = 8424 bits + CRC24A);
E follows from the configured rate and modulation and fills the data
REs of as many slots as it needs, blocks back to back. The receiver
gets no genie: timing, CFO, slot identity, channel and noise variance
are all estimated from the waveform and the DM-RS. Timing is
deliberately backed off into the CP, because sampling early is a
phase ramp the channel estimator absorbs while sampling late is ISI;
the DM-RS detector and the DFT denoiser are both built
ramp-invariant for that reason. Not modelled: multiple code blocks
per transport block and the other BG1 lifting sizes, HARQ redundancy
versions beyond rv0, MIMO and transform precoding, PT-RS (the
decision-directed CPE stands in), sampling clock offset, and the
SSB/PRACH acquisition procedures. The natural extension points are
noted in the module docstrings.

## References

* 3GPP TS 38.211: physical channels and modulation (numerology, QAM,
  DM-RS, Gold sequences)
* 3GPP TS 38.212: multiplexing and channel coding (LDPC, CRC, rate
  matching)
* 3GPP TR 38.901: channel models (TDL profiles)
* 3GPP TS 38.101-1: UE transmit EVM limits
* J. van de Beek, M. Sandell, P. Borjesson, "ML estimation of time
  and frequency offset in OFDM systems," IEEE Trans. Signal Process.,
  Jul. 1997
* M. Speth, S. Fechtel, G. Fock, H. Meyr, "Optimum receiver design
  for wireless broad-band systems using OFDM," IEEE Trans. Commun.,
  Nov. 1999

MIT licence. If you spot a deviation from the standard, open an
issue; the structure tests in `tests/` are the place to encode it.

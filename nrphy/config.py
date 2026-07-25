"""Every knob of the simulation, with defaults that make a 20 MHz
5G NR carrier at 30 kHz subcarrier spacing: 51 resource blocks, a
1024-point FFT at 30.72 Msps, normal cyclic prefix, one transport
block per run of consecutive slots. Set channel="tdl-c" with a
Doppler for the fading story; "awgn" is the calibration anchor.
"""

from dataclasses import dataclass


@dataclass
class SimConfig:
    # numerology (TS 38.211)
    scs_khz: float = 30.0          # subcarrier spacing, mu = 1
    n_rb: int = 51                 # resource blocks (12 subcarriers each)
    n_fft: int = 1024
    # transport (TS 38.212)
    qam: int = 16                  # 4 | 16 | 64 | 256
    code_rate: float = 0.5         # LDPC BG1, K = 8448, E from this
    n_tb: int = 8                  # transport blocks per run
    ldpc_iters: int = 40
    # pilots (TS 38.211 type-1 DM-RS, comb-2)
    dmrs_symbols: tuple = (2, 11)  # positions within the 14-symbol slot
    n_id: int = 41                 # scrambling identity
    # transmitter front end
    window_samples: int = 18       # WOLA edge, spent from the CP; 0 = plain
    os: int = 2                    # DAC oversampling for images and PA
    dac_bits: int = 10
    tx_iq_amp_db: float = 0.2      # I/Q modulator imbalance
    tx_iq_phase_deg: float = 1.0
    tx_lo_leak_db: float = -35.0   # carrier feedthrough
    pa_ibo_db: float = 8.0         # Rapp PA input backoff from saturation
    pa_p: float = 2.0              # Rapp smoothness; 0 disables the PA
    # channel
    channel: str = "awgn"          # "awgn" | "tdl-a" | "tdl-b" | "tdl-c"
    delay_spread_ns: float = 300.0
    doppler_hz: float = 70.0       # max Doppler of the Jakes spectrum
    snr_db: float = 12.0           # Es/N0 per occupied subcarrier
    cfo_hz: float = 41000.0        # carrier frequency offset (1.4 SCS)
    # receiver front end
    rx_iq_amp_db: float = 0.2
    rx_iq_phase_deg: float = 1.0
    adc_bits: int = 10
    # receiver DSP
    chanest: str = "dft"           # "ls-linear" | "dft" (denoised)
    cpe_track: bool = True         # per-symbol common phase from decisions
    # bookkeeping
    include_impairments: bool = True

    @property
    def fs(self):
        return self.n_fft * self.scs_khz * 1e3

    @property
    def n_sc(self):
        return 12 * self.n_rb

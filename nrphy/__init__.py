"""nrphy: baseband PHY simulation of a 5G NR OFDM transceiver."""

from .config import SimConfig
from .sim import run_link

__version__ = "0.1.0"
__all__ = ["SimConfig", "run_link"]

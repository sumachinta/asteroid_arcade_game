import numpy as np
from dataclasses import dataclass
from typing import Dict

from utils.encoding import StimFreqs


@dataclass
class StimConfig:
    """
    Parameters for generating biphasic stimulation waveforms.
    All times in seconds, amplitude in Volts (75 mV = 0.075).
    """
    sampling_rate: int = 20_000         # samples per second
    pulse_amplitude: float = 0.075      # 75 mV biphasic
    phase_width_s: float = 200e-6       # 200 µs per phase
    inter_phase_gap_s: float = 0.0      # gap between phases (usually 0)

    @property
    def dt(self) -> float:
        return 1.0 / self.sampling_rate


# -----------------------------
# 2. Single biphasic pulse
# -----------------------------

def make_biphasic_pulse(cfg: StimConfig) -> np.ndarray:
    """
    Create a single biphasic pulse (+amp then -amp) at the given sampling rate.
    Returns: 1D numpy array of shape (n_samples_in_pulse,).
    """
    n_phase = int(round(cfg.phase_width_s * cfg.sampling_rate))
    n_gap   = int(round(cfg.inter_phase_gap_s * cfg.sampling_rate))

    phase1 = np.full(n_phase,  cfg.pulse_amplitude, dtype=float)
    gap    = np.zeros(n_gap, dtype=float) if n_gap > 0 else np.array([], dtype=float)
    phase2 = np.full(n_phase, -cfg.pulse_amplitude, dtype=float)

    pulse = np.concatenate([phase1, gap, phase2])
    return pulse


# ---------------------------------------------
# 3. Constant-frequency pulse train (1 channel)
# ---------------------------------------------

def generate_pulse_train_constant_freq(freq_hz: float,
                                       duration_s: float,
                                       cfg: StimConfig) -> np.ndarray:
    """
    Generate a biphasic pulse train at a constant frequency over `duration_s`.
    No jitter, perfectly periodic pulses.
    """
    total_samples = int(round(duration_s * cfg.sampling_rate))
    signal = np.zeros(total_samples, dtype=float)

    if freq_hz <= 0.0:
        return signal  # no stim

    pulse = make_biphasic_pulse(cfg)
    pulse_len = len(pulse)

    period_s = 1.0 / freq_hz
    period_samples = int(round(period_s * cfg.sampling_rate))

    t = 0
    while t + pulse_len <= total_samples:
        signal[t:t + pulse_len] += pulse
        t += period_samples

    return signal


# --------------------------------------------------------
# 4. Pulse trains for left / center / right using StimFrequencies
# --------------------------------------------------------

@dataclass
class DirectionalWaveforms:
    left: np.ndarray
    center: np.ndarray
    right: np.ndarray

def generate_directional_trains(freqs: StimFreqs,
                                           duration_s: float,
                                           cfg: StimConfig) -> DirectionalWaveforms:
    """
    Generate pulse trains for left, center, and right directions using
    the StimFrequencies dataclass.
    """
    left_wave = generate_pulse_train_constant_freq(freqs.left_hz,   duration_s, cfg)
    center_wave = generate_pulse_train_constant_freq(freqs.center_hz, duration_s, cfg)
    right_wave = generate_pulse_train_constant_freq(freqs.right_hz, duration_s, cfg)

    return DirectionalWaveforms(
        left=left_wave,
        center=center_wave,
        right=right_wave,
    )
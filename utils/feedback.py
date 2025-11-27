from dataclasses import dataclass
from typing import List, Tuple
from enum import Enum, auto
from dataclasses import replace
import random
from utils.stimulation import StimConfig, StimFreqs, generate_directional_trains
import numpy as np

# Choose feedback loop here:
#   FeedbackMode.LOOP1  -> basic (hit punish, kill reward)
#   FeedbackMode.LOOP2  -> adds survival reward
FEEDBACK_MODE_DEFAULT = "LOOP1"  # change to "LOOP2" to switch

@dataclass
class FeedbackState:
    survival_timer_s: float = 0.0


class FeedbackMode(Enum):
    LOOP1 = auto()
    LOOP2 = auto()


class FeedbackType(Enum):
    NONE = auto()
    PUNISHMENT = auto()
    KILL_REWARD = auto()
    SURVIVAL_REWARD = auto()

#  Feedback stimulation parameters 
# Reward (kill/survival) – short, strong, high-freq, all encoding electrodes
FEEDBACK_REWARD_FREQ_HZ = 100.0    # Hz
FEEDBACK_REWARD_DURATION_S = 0.100 # 100 ms
FEEDBACK_REWARD_AMP_V = 0.075      # 75 mV (same as normal sensory stim, or tune)

# Punishment – long, low-freq, higher amplitude, random encoding electrode
FEEDBACK_PUNISH_FREQ_HZ = 5.0      # Hz
FEEDBACK_PUNISH_DURATION_S = 4.0   # 4 s of punishment
FEEDBACK_PUNISH_AMP_V = 0.150      # 150 mV


def step_feedback(
    mode: FeedbackMode,
    fb_state: FeedbackState,
    hit: bool,
    kill: bool,
    dt: float,
    punishment_total_pause_s: float = 8.0,  # 4 s stim + 4 s silence
    reward_pause_s: float = 0.1,            # 100 ms
    survival_threshold_s: float = 1.0,      # for loop 2
) -> Tuple[FeedbackType, float, bool]:
    """
    Returns (feedback_type, pause_sensory_s, reset_game).
    """
    # Loop 1: hit -> punishment; kill -> reward; survival implicit
    if mode == FeedbackMode.LOOP1:
        if hit:
            fb_state.survival_timer_s = 0.0
            return FeedbackType.PUNISHMENT, punishment_total_pause_s, True
        if kill:
            fb_state.survival_timer_s = 0.0
            return FeedbackType.KILL_REWARD, reward_pause_s, False

        fb_state.survival_timer_s += dt
        return FeedbackType.NONE, 0.0, False

    # Loop 2: hit -> punishment; kill -> reward; survival reward after window
    if hit:
        fb_state.survival_timer_s = 0.0
        return FeedbackType.PUNISHMENT, punishment_total_pause_s, True

    if kill:
        fb_state.survival_timer_s = 0.0
        return FeedbackType.KILL_REWARD, reward_pause_s, False

    fb_state.survival_timer_s += dt
    if fb_state.survival_timer_s >= survival_threshold_s:
        fb_state.survival_timer_s = 0.0
        return FeedbackType.SURVIVAL_REWARD, reward_pause_s, False

    return FeedbackType.NONE, 0.0, False


def generate_feedback_trains(
    fb_type: FeedbackType,
    base_cfg: StimConfig,
) -> tuple[StimFreqs, dict[str, np.ndarray], float, StimConfig]:
    """
    Generate stimulation trains for a feedback event, using the same
    directional stim infrastructure as sensory encoding.

    Returns:
        (stim_freqs, stim_trains, duration_s, cfg_used)

        stim_freqs : StimFreqs used for this feedback
        stim_trains: dict with "left", "center", "right" waveform arrays
        duration_s : duration of the feedback train
        cfg_used   : StimConfig actually used (may differ in amplitude)
    """
    # Kill / survival reward: high-freq burst on all encoding electrodes 
    if fb_type in (FeedbackType.KILL_REWARD, FeedbackType.SURVIVAL_REWARD):
        duration_s = FEEDBACK_REWARD_DURATION_S
        cfg_used = replace(base_cfg, pulse_amplitude=FEEDBACK_REWARD_AMP_V)

        stim_freqs = StimFreqs(
            left_hz=FEEDBACK_REWARD_FREQ_HZ,
            center_hz=FEEDBACK_REWARD_FREQ_HZ,
            right_hz=FEEDBACK_REWARD_FREQ_HZ,
        )

        stim_trains = generate_directional_trains(stim_freqs, duration_s, cfg_used)
        return stim_freqs, stim_trains, duration_s, cfg_used

    # Punishment: low-freq, high-amp on a all encoding electrode 
    elif fb_type == FeedbackType.PUNISHMENT:
        duration_s = FEEDBACK_PUNISH_DURATION_S
        cfg_used = replace(base_cfg, pulse_amplitude=FEEDBACK_PUNISH_AMP_V)

        left_hz   = FEEDBACK_PUNISH_FREQ_HZ 
        center_hz = FEEDBACK_PUNISH_FREQ_HZ 
        right_hz  = FEEDBACK_PUNISH_FREQ_HZ 

        stim_freqs = StimFreqs(left_hz=left_hz,center_hz=center_hz,right_hz=right_hz)
        stim_trains = generate_directional_trains(stim_freqs, duration_s, cfg_used)
        return stim_freqs, stim_trains, duration_s, cfg_used

    else:
        raise ValueError(f"Unsupported feedback type for trains: {fb_type}")

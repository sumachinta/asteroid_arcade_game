from dataclasses import dataclass
from typing import List, Tuple
from enum import Enum, auto


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

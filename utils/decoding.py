from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Heading(Enum):
    LEFT = "left"
    RIGHT = "right"
    NONE = "none"  # no turn


@dataclass
class FiringCounts:
    """
    Spike counts in one 10 ms bin for each decoding group.
    """
    left: int     # heading: turn left
    right: int    # heading: turn right
    thrust: int   # accelerate
    shoot: int    # fire


@dataclass
class FiringRates:
    """
    Firing rates (spikes/s) for each decoding group, in one bin.
    """
    left: float
    right: float
    thrust: float
    shoot: float


@dataclass
class DecodingConfig:
    """
    Thresholds and timing for decoding.
    Adjust these empirically later.
    """
    bin_duration_s: float = 0.010  # 10 ms

    # Heading (turn) thresholds
    heading_silence_threshold: float = 5.0   # spikes/s total (left + right)
    heading_diff_threshold: float = 3.0      # spikes/s difference required to pick a side

    # Thrust (accelerate) threshold
    thrust_threshold: float = 5.0            # spikes/s

    # Shoot (fire) threshold + cooldown
    shoot_threshold: float = 8.0             # spikes/s
    shoot_cooldown_s: float = 0.150         # min time between shots in seconds


@dataclass
class Action:
    """
    Decoded action for one 10 ms time step.
    """
    heading: Heading     # LEFT / RIGHT / NONE
    thrust_on: bool      # True = accelerate
    shoot: bool          # True = fire a bullet


@dataclass
class DecoderState:
    """
    Stateful info for the decoder (e.g., last shot time).
    """
    last_shot_time_s: float = -1e9  # effectively "never shot"


def counts_to_rates(counts: FiringCounts, cfg: DecodingConfig) -> FiringRates:
    """
    Convert spike counts in a bin to firing rates in spikes/s.
    """
    dt = cfg.bin_duration_s
    return FiringRates(
        left   = counts.left / dt,
        right  = counts.right / dt,
        thrust = counts.thrust / dt,
        shoot  = counts.shoot / dt,
    )


class NeuralDecoder:
    """
    Decodes neural firing into game actions at each time step (bin).
    """

    def __init__(self, config: Optional[DecodingConfig] = None):
        self.cfg = config or DecodingConfig()
        self.state = DecoderState()

    # Heading (turn left/right/none) 

    def decode_heading(self, r_left: float, r_right: float) -> Heading:
        """
        Winner-take-all between left and right, with silence and difference thresholds.
        """
        cfg = self.cfg
        r_total = r_left + r_right

        # Not enough activity to bother turning
        if r_total < cfg.heading_silence_threshold:
            return Heading.NONE

        diff = r_left - r_right

        if diff > cfg.heading_diff_threshold:
            return Heading.LEFT
        elif diff < -cfg.heading_diff_threshold:
            return Heading.RIGHT
        else:
            # Similar activity: don't twitch for tiny differences
            return Heading.NONE

    # Thrust (accelerate or not) 

    def decode_thrust(self, r_thrust: float) -> bool:
        """
        Binary decision for acceleration based on a single group.
        """
        return r_thrust > self.cfg.thrust_threshold

    # Shoot (fire or not, with cooldown) 

    def decode_shoot(self, r_shoot: float, t_s: float) -> bool:
        """
        Binary shoot decision with a cooldown to avoid continuous firing.
        """
        cfg = self.cfg

        # Basic threshold
        if r_shoot <= cfg.shoot_threshold:
            return False

        # Check cooldown
        if (t_s - self.state.last_shot_time_s) < cfg.shoot_cooldown_s:
            return False

        # Approve a shot and update last_shot_time
        self.state.last_shot_time_s = t_s
        return True

    # 1 full decoding step

    def step(self, counts: FiringCounts, t_s: float) -> Action:
        """
        Decode a single time-bin of spikes into an Action.
        counts: FiringCounts for this bin.
        t_s: absolute time (seconds) at the *end* of this bin.
        """
        rates = counts_to_rates(counts, self.cfg)

        heading = self.decode_heading(rates.left, rates.right)
        thrust_on = self.decode_thrust(rates.thrust)
        shoot = self.decode_shoot(rates.shoot, t_s)

        return Action(heading=heading, thrust_on=thrust_on, shoot=shoot)


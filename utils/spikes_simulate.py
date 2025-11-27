import numpy as np
from dataclasses import dataclass
from typing import List
from math import floor

@dataclass
class RandomRateRanges:
    """
    Defines min–max Hz for each decoding group.
    """
    left_range:   tuple = (2, 50)
    right_range:  tuple = (2, 50)
    thrust_range: tuple = (2, 50)
    shoot_range:  tuple = (5, 50)

@dataclass
class FiringCounts:
    """
    Spike counts in one 10 ms bin for each decoding group.
    """
    left: int     # heading: turn left
    right: int    # heading: turn right
    thrust: int   # accelerate
    shoot: int    # fire


def pick_random_rates(ranges: RandomRateRanges) -> dict:
    """
    Randomly sample a firing rate for each neural group within the given ranges.
    Returns a dict: {"left": hz, "right": hz, "thrust": hz, "shoot": hz}
    """
    return {
        "left":   np.random.uniform(*ranges.left_range),
        "right":  np.random.uniform(*ranges.right_range),
        "thrust": np.random.uniform(*ranges.thrust_range),
        "shoot":  np.random.uniform(*ranges.shoot_range),
    }


def simulate_step_firing_counts(ranges: RandomRateRanges = RandomRateRanges(),
                                bin_duration_s: float = 0.010) -> FiringCounts:
    """
    Simulate Poisson spike COUNTS for a single 10 ms bin, using
    random firing rates chosen within biologically plausible ranges.
    This is equivalent to a 20 kHz Poisson train binned to 10 ms.
    """
    rates = pick_random_rates(ranges)          # Hz
   
    def poisson_spikes(rate_hz):
         # Poisson mean = rate * bin_duration
        return int(np.random.poisson(rate_hz * bin_duration_s))

    return FiringCounts(
        left=poisson_spikes(rates["left"]),
        right=poisson_spikes(rates["right"]),
        thrust=poisson_spikes(rates["thrust"]),
        shoot=poisson_spikes(rates["shoot"]),
    )


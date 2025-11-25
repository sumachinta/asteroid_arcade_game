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

@dataclass
class SimSpikeTrain:
    left: np.ndarray
    right: np.ndarray
    thrust: np.ndarray
    shoot: np.ndarray
    rates: dict = None  # store the actual firing rates used


def simulate_spike_trains(duration_s: float,
                                 ranges: RandomRateRanges,
                                 sampling_rate: int = 20_000) -> SimSpikeTrain:
    """
    1. Randomly picks Poisson rates within given ranges
    2. Generates spike trains for left/right/thrust/shoot groups.
    """
    # Step 1 — Sample random firing rates
    rates = pick_random_rates(ranges)

    n_samples = int(duration_s * sampling_rate)
    dt = 1.0 / sampling_rate

    # Poisson spikes: P(spike) = rate * dt
    def gen(rate):
        p = rate * dt
        return (np.random.rand(n_samples) < p).astype(int)

    return SimSpikeTrain(
        left   = gen(rates["left"]),
        right  = gen(rates["right"]),
        thrust = gen(rates["thrust"]),
        shoot  = gen(rates["shoot"]),
        rates  = rates,
    )


@dataclass
class FiringCounts:
    """
    Spike counts in one 10 ms bin for each decoding group.
    """
    left: int     # heading: turn left
    right: int    # heading: turn right
    thrust: int   # accelerate
    shoot: int    # fire


def bin_spike_trains(trains: SimSpikeTrain,
                    bin_size_s: float = 0.010,
                    sampling_rate: int = 20_000) -> List[FiringCounts]:
    """
    Bin the simulated spike trains into 10 ms firing counts.
    """
    bin_size_samples = int(bin_size_s * sampling_rate)
    n_samples = len(trains.left)
    n_bins = n_samples // bin_size_samples

    binned = []

    for b in range(n_bins):
        start = b * bin_size_samples
        end   = start + bin_size_samples

        counts = FiringCounts(
            left   = int(trains.left[start:end].sum()),
            right  = int(trains.right[start:end].sum()),
            thrust = int(trains.thrust[start:end].sum()),
            shoot  = int(trains.shoot[start:end].sum()),
        )
        binned.append(counts)

    return binned

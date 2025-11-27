import math
from typing import List, Dict, Tuple
from dataclasses import dataclass
from utils.game_physics import Ship, Asteroid, Bullet
from utils.game_physics import wrap_angle

MAX_DISTANCE = 400.0  # for threat computation
DT = 0.010  # 10 ms bin / game step

F_MIN_Hz = 5.0 # minimum frequency for stimulation
F_MAX_Hz = 50.0 # maximum frequency for stimulation

@dataclass
class Threat:
    left: float # Expected to be in [0, 1]
    center: float 
    right: float 

@dataclass
class StimFreqs:
    left_hz: float  # in Hz
    center_hz: float  
    right_hz: float  



def compute_directional_threat(
    ship: Ship,
    asteroids: list[Asteroid],
    max_size: float,
    max_dist: float = MAX_DISTANCE,
    theta_center_deg: float = 10.0,
    w_dist: float = 0.5,
    w_speed: float = 0.3,
    w_size: float = 0.2,
) -> Threat:
    """
    Compute maximum threat from left, center, right directions
    Returns Threat(left, center, right), each in [0, 1]
    """

    def distance_threat(distance: float) -> float:
        """
        Threat component based purely on distance.
        Closer asteroid → value closer to 1. At max_dist → 0.
        """
        if distance >= max_dist:
            return 0.0
        # linear falloff: 0 at max_dist, 1 at distance=0
        return max(0.0, 1.0 - distance / max_dist)

    def speed_threat(rel_vx: float,rel_vy: float,
        unit_rx: float,unit_ry: float,
        speed_scale: float = 200.0) -> float:
        """
        Threat component based on closing speed (how fast it is coming toward the ship).
        Only approaching motion contributes (moving away → 0).
        """
        # Relative velocity projected onto line-of-sight (ship -> asteroid)
        # Negative dot product means "moving toward ship", hence minus sign
        closing_speed = -(rel_vx * unit_rx + rel_vy * unit_ry)

        # Ignore receding asteroids
        if closing_speed <= 0:
            return 0.0
        return min(1.0, closing_speed / speed_scale)

    def size_threat(size: float) -> float:
        """
        Threat component based on asteroid size
        max_size → 1.0, smaller sizes scale linearly down to 0
        """
        if max_size <= 0:
            return 0.0
        return min(1.0, size / max_size)


    center_half_angle = math.radians(theta_center_deg)

    threat_left = 0.0
    threat_center = 0.0
    threat_right = 0.0

    for asteroid in asteroids:
        # relative position & distance to ship
        rel_x = asteroid.x - ship.x
        rel_y = asteroid.y - ship.y
        distance = math.hypot(rel_x, rel_y)  # distance ship <-> asteroid

        # Skip if sitting exactly on the ship or too far away to matter
        if distance == 0.0 or distance > max_dist:
            continue

        # Unit vector from ship to asteroid
        unit_rx = rel_x / distance
        unit_ry = rel_y / distance

        abs_angle_to_asteroid = math.atan2(rel_y, rel_x)
        rel_angle_to_asteroid = wrap_angle(abs_angle_to_asteroid - ship.heading)

        # Threat components 
        dist_component = distance_threat(distance)

        rel_vx = asteroid.vx - ship.vx
        rel_vy = asteroid.vy - ship.vy
        speed_component = speed_threat(rel_vx, rel_vy, unit_rx, unit_ry)

        size_component = size_threat(asteroid.size)

        # Combine components into a single per-asteroid threat 
        raw_threat = (w_dist * dist_component + w_speed * speed_component+ w_size * size_component)
        per_asteroid_threat = max(0.0, min(1.0, raw_threat))

        # Put threat into left / center / right sector based on relative angle
        if rel_angle_to_asteroid < -center_half_angle:
            threat_right = max(threat_right, per_asteroid_threat)
        elif rel_angle_to_asteroid > center_half_angle:
            threat_left = max(threat_left, per_asteroid_threat)
        else:
            threat_center = max(threat_center, per_asteroid_threat)
    return Threat(
        left=round(threat_left,2), 
        center=round(threat_center,2), 
        right=round(threat_right,2))

    
        
def map_threat_to_stim_freqs(
    threat: Threat,
    f_min_hz: float = F_MIN_Hz,
    f_max_hz: float = F_MAX_Hz,
) -> StimFreqs:
    """
    Map threat values in [0, 1] to stimulation frequencies in [f_min_hz, f_max_hz].
    Higher threat → higher frequency.
    """
    def map_value(threat_value: float) -> float:
        return f_min_hz + threat_value * (f_max_hz - f_min_hz)

    left_hz = round(map_value(threat.left))
    center_hz = round(map_value(threat.center))
    right_hz = round(map_value(threat.right))

    return StimFreqs(left_hz=left_hz, center_hz=center_hz, right_hz=right_hz)
import math
from typing import List, Dict, Tuple
from dataclasses import dataclass


WORLD_WIDTH  = 800   
WORLD_HEIGHT = 600

F_MIN_Hz = 5.0 # minimum frequency for stimulation
F_MAX_Hz = 50.0 # maximum frequency for stimulation


@dataclass
class Ship:
    x: float
    y: float
    vx: float
    vy: float
    heading: float  # in radians

@dataclass
class Asteroid:
    x: float
    y: float
    vx: float
    vy: float
    size: float

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

@dataclass
class GameState:
    ship: Ship
    asteroids: List[Asteroid]
    t_s: float = 0.0


def wrap_angle(angle: float) -> float:
    """
    Map any angle to the range [-pi, pi].
    """
    return (angle + math.pi) % (2 * math.pi) - math.pi

def wrap_position(x: float, y: float) -> tuple[float, float]:
    x = x % WORLD_WIDTH
    y = y % WORLD_HEIGHT
    return x, y

def compute_directional_threat(
    ship: Ship,
    asteroids: list[Asteroid],
    max_dist: float,
    max_size: float,
    theta_center_deg: float = 10.0,
    w_dist: float = 0.5,
    w_speed: float = 0.3,
    w_size: float = 0.2,
) -> Threat:
    """
    Compute maximum threat coming from the left, center, and right directions.
    Returns:
        (threat_left, threat_center, threat_right), each in [0, 1].
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

    def speed_threat(
        rel_vx: float,
        rel_vy: float,
        unit_rx: float,
        unit_ry: float,
        speed_scale: float = 200.0,
    ) -> float:
        """
        Threat component based on closing speed (how fast it is coming toward the ship).
        Only approaching motion contributes (moving away → 0).
        """
        # Relative velocity projected onto line-of-sight (ship -> asteroid)
        # Negative dot product means "moving toward ship", hence minus sign.
        closing_speed = -(rel_vx * unit_rx + rel_vy * unit_ry)

        # Ignore receding asteroids
        if closing_speed <= 0:
            return 0.0

        # Map closing speed into [0, 1] using a simple scaling
        return min(1.0, closing_speed / speed_scale)

    def size_threat(size: float) -> float:
        """
        Threat component based on asteroid size (radius).
        max_size → 1.0, smaller sizes scale linearly.
        """
        if max_size <= 0:
            return 0.0
        return min(1.0, size / max_size)


    # Convert center sector half-angle to radians
    center_half_angle = math.radians(theta_center_deg)

    # Initialize directional threats (winner-take-all max per sector)
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

        # Relative angle: which side of the ship? 
        # Absolute angle of asteroid in world coordinates
        abs_angle_to_asteroid = math.atan2(rel_y, rel_x)

        # Angle of asteroid relative to ship's heading (egocentric)
        rel_angle_to_asteroid = wrap_angle(abs_angle_to_asteroid - ship.heading)

        # Threat components 
        # (a) distance-based component
        dist_component = distance_threat(distance)

        # (b) speed-based component (uses relative velocity)
        rel_vx = asteroid.vx - ship.vx
        rel_vy = asteroid.vy - ship.vy
        speed_component = speed_threat(rel_vx, rel_vy, unit_rx, unit_ry)

        # (c) size-based component
        size_component = size_threat(asteroid.size)

        # Combine components into a single per-asteroid threat 
        raw_threat = (w_dist * dist_component + w_speed * speed_component+ w_size * size_component)

        # Clamp final threat to [0, 1]
        per_asteroid_threat = max(0.0, min(1.0, raw_threat))

        # Put threat into left / center / right sector based on relative angle
        if rel_angle_to_asteroid < -center_half_angle:
            threat_right = max(threat_right, per_asteroid_threat)
        elif rel_angle_to_asteroid > center_half_angle:
            threat_left = max(threat_left, per_asteroid_threat)
        else:
            threat_center = max(threat_center, per_asteroid_threat)
    return Threat(left=round(threat_left,2), center=round(threat_center,2), right=round(threat_right,2))


def update_ship(
    ship: Ship,
    action,          # your Action dataclass from decoder
    dt: float,
    turn_rate_rad_s: float = math.radians(180),  # 180°/s
    thrust_accel: float = 50.0,
):
    # Turn
    if action.heading.value == "left":
        ship.heading -= turn_rate_rad_s * dt
    elif action.heading.value == "right":
        ship.heading += turn_rate_rad_s * dt

    ship.heading = wrap_angle(ship.heading)

    # Thrust
    if action.thrust_on:
        ax = thrust_accel * math.cos(ship.heading)
        ay = thrust_accel * math.sin(ship.heading)
    else:
        ax = ay = 0.0

    # Integrate velocity and position
    ship.vx += ax * dt
    ship.vy += ay * dt

    ship.x += ship.vx * dt
    ship.y += ship.vy * dt

    ship.x, ship.y = wrap_position(ship.x, ship.y)

def update_asteroids(asteroids: List[Asteroid], dt: float):
    for a in asteroids:
        a.x += a.vx * dt
        a.y += a.vy * dt
        a.x, a.y = wrap_position(a.x, a.y)


def update_game_state(state: GameState, action, dt: float):
    update_ship(state.ship, action, dt)
    update_asteroids(state.asteroids, dt)
    state.t_s += dt
    
        
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
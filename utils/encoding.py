import math
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class Ship:
    x: float
    y: float
    vx: float
    vy: float
    heading: float  # in radians

    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)
    
    def velocity(self) -> Tuple[float, float]:
        return (self.vx, self.vy)
    
    def update_position(self, dt: float = 0.01) -> None:
        # Update position based on velocity and time step
        self.x += self.vx * dt
        self.y += self.vy * dt

    def set_heading(self, angle: float) -> None:
        self.heading = angle


@dataclass
class Asteroid:
    x: float
    y: float
    vx: float
    vy: float
    size: float

    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)
    
    def velocity(self) -> Tuple[float, float]:
        return (self.vx, self.vy)
    
    def update_position(self, dt: float = 0.01) -> None:
        # Update position based on velocity and time step
        self.x += self.vx * dt
        self.y += self.vy * dt


def wrap_angle(angle: float) -> float:
    """
    Map any angle to the range [-pi, pi].
    """
    return (angle + math.pi) % (2 * math.pi) - math.pi


def compute_directional_threat(
    ship: Ship,
    asteroids: list[Asteroid],
    max_dist: float,
    max_size: float,
    theta_center_deg: float = 30.0,
    w_dist: float = 0.5,
    w_speed: float = 0.3,
    w_size: float = 0.2,
):
    theta_c = math.radians(theta_center_deg)

    d_left = d_center = d_right = 0.0

    for ast in asteroids:

        dx = ast.x - ship.x
        dy = ast.y - ship.y
        dist = math.hypot(dx, dy)
        if dist == 0 or dist > max_dist:
            continue

        angle_to_ast = math.atan2(dy, dx)
        angle_diff = wrap_angle(angle_to_ast - ship.heading)

        # Distance term
        dist_term = max(0.0, 1.0 - dist / max_dist)

        # Closing speed term
        vx_rel = ast.vx - ship.vx
        vy_rel = ast.vy - ship.vy
        ux, uy = dx / dist, dy / dist
        closing_speed = -(vx_rel * ux + vy_rel * uy)
        closing_speed = max(0.0, closing_speed)
        speed_term = min(1.0, closing_speed / 200.0)

        # Size term
        size_term = min(1.0, ast.size / max_size)

        # Combined threat
        threat = w_dist * dist_term + w_speed * speed_term + w_size * size_term
        threat = max(0.0, min(1.0, threat))

        # Assign directional bins
        if angle_diff < -theta_c:
            d_left   = max(d_left, threat)
        elif angle_diff > +theta_c:
            d_right  = max(d_right, threat)
        else:
            d_center = max(d_center, threat)

    return d_left, d_center, d_right


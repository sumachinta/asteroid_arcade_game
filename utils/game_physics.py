from typing import List, Tuple
from dataclasses import dataclass
import math
from utils.decoding import Action

WORLD_WIDTH  = 800   
WORLD_HEIGHT = 600

SHIP_RADIUS = 10.0
BULLET_RADIUS = 3.0
BULLET_MAX_AGE_S = 3.0  # seconds
DT = 0.010  # 10 ms bin / game step

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
    alive: bool = True

@dataclass
class Bullet:
    x: float
    y: float
    vx: float
    vy: float
    alive: bool = True
    age_s: float = 0.0  # how long the bullet has existed

@dataclass
class GameState:
    ship: Ship
    asteroids: List[Asteroid]
    bullets: List[Bullet]
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

def circle_collision(x1, y1, r1, x2, y2, r2) -> bool:
    """checks whether two circles overlap"""
    dx = x1 - x2
    dy = y1 - y2
    return dx * dx + dy * dy <= (r1 + r2) * (r1 + r2)


def spawn_bullet(ship: Ship, bullet_speed: float = 200.0) -> Bullet:
    """ Create a bullet at the ship's position, moving in the ship's heading direction."""
    vx = bullet_speed * math.cos(ship.heading)
    vy = bullet_speed * math.sin(ship.heading)
    return Bullet(x=ship.x, y=ship.y, vx=vx, vy=vy, alive=True, age_s=0.0)

def update_ship(ship: Ship, action: Action, dt: float,
                turn_rate_rad_s: float = math.radians(180),  # 180°/s
                thrust_accel: float = 50.0):
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
        if not a.alive:
            continue
        a.x += a.vx * dt
        a.y += a.vy * dt
        a.x, a.y = wrap_position(a.x, a.y)
    asteroids[:] = [a for a in asteroids if a.alive]

def update_bullets(bullets: List[Bullet], dt: float):
    """
    Move bullets, wrap positions, age them, and mark as dead
    once they exceed BULLET_MAX_AGE_S.
    """
    for b in bullets:
        if not b.alive:
            continue
        b.x += b.vx * dt
        b.y += b.vy * dt
        b.x, b.y = wrap_position(b.x, b.y)

        # Age bullet and kill after max lifetime
        b.age_s += dt
        if b.age_s >= BULLET_MAX_AGE_S:
            b.alive = False
    bullets[:] = [b for b in bullets if b.alive]


def detect_hits_and_kills(state: GameState) -> Tuple[bool, bool]:
    hit = False
    kill = False

    # Ship–asteroid collisions
    for a in state.asteroids:
        if not a.alive:
            continue
        if circle_collision(state.ship.x, state.ship.y, SHIP_RADIUS, a.x, a.y, a.size):
            hit = True
            break

    # Bullet–asteroid collisions
    for b in state.bullets:
        if not b.alive:
            continue
        for a in state.asteroids:
            if not a.alive:
                continue
            if circle_collision(b.x, b.y, BULLET_RADIUS, a.x, a.y, a.size):
                kill = True
                a.alive = False
                b.alive = False
                break
            
    state.asteroids = [a for a in state.asteroids if a.alive]
    state.bullets   = [b for b in state.bullets   if b.alive]
    return hit, kill

def update_game_state(state: GameState, action: Action, dt: float):
    """ Update the game state by one time step given the action"""
    if action.shoot:
        state.bullets.append(spawn_bullet(state.ship))
    update_ship(state.ship, action, dt)
    update_asteroids(state.asteroids, dt)
    update_bullets(state.bullets, dt)
    hit, kill = detect_hits_and_kills(state)
    state.t_s += dt
    return hit, kill




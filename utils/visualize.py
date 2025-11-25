import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple, List

from utils.encoding import Ship, Asteroid

# assuming these are already defined somewhere:
# from your_module import Ship, Asteroid

def visualize_game(
    ship: "Ship",
    asteroids: List["Asteroid"],
    d_threat: Optional[Tuple[float, float, float]] = None,
    max_dist: float = 400.0,
    theta_center_deg: float = 30.0,
):
    """
    Visualize the ship, asteroids, and optional directional threats.

    ship      : Ship instance (x, y, heading used here)
    asteroids : list of Asteroid instances (x, y, size used here)
    d_threat  : optional (d_left, d_center, d_right) to display on the plot
    """

    x_s, y_s = ship.x, ship.y
    heading = ship.heading

    fig, ax = plt.subplots(figsize=(7, 7))

    # --- Ship triangle ---
    def ship_triangle(x, y, angle, scale=20):
        # Triangle in local (ship) coordinates
        pts = np.array([[1, 0], [-0.5, 0.5], [-0.5, -0.5]]) * scale
        rot = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle),  np.cos(angle)]])
        pts = pts @ rot.T
        pts[:, 0] += x
        pts[:, 1] += y
        return pts

    tri = ship_triangle(x_s, y_s, heading)
    ax.fill(tri[:, 0], tri[:, 1], alpha=0.7, label="Ship")

    # --- Asteroids ---
    for ast in asteroids:
        ax.add_patch(
            plt.Circle((ast.x, ast.y), ast.size, alpha=0.6)
        )

    # --- Direction sectors (left / center / right) ---
    theta_c = np.radians(theta_center_deg)

    # center cone arc
    cone_angles = np.linspace(-theta_c, theta_c, 40)
    cone_x = x_s + max_dist * np.cos(cone_angles + heading)
    cone_y = y_s + max_dist * np.sin(cone_angles + heading)
    ax.plot(cone_x, cone_y, ls='--', label="Center sector")

    # left boundary
    left_angle = heading - theta_c
    lx = x_s + max_dist * np.cos(left_angle)
    ly = y_s + max_dist * np.sin(left_angle)
    ax.plot([x_s, lx], [y_s, ly], ls='--')

    # right boundary
    right_angle = heading + theta_c
    rx = x_s + max_dist * np.cos(right_angle)
    ry = y_s + max_dist * np.sin(right_angle)
    ax.plot([x_s, rx], [y_s, ry], ls='--')

    # --- Optional threat text ---
    if d_threat is not None:
        d_left, d_center, d_right = d_threat
        text = (
            f"Threats:\n"
            f"Left   = {d_left:.2f}\n"
            f"Center = {d_center:.2f}\n"
            f"Right  = {d_right:.2f}"
        )
        ax.text(
            0.02, 0.98, text,
            transform=ax.transAxes,
            fontsize=12,
            va='top',
            bbox=dict(facecolor='white', alpha=0.7)
        )

    # --- Formatting ---
    ax.set_aspect('equal')
    ax.set_title("Asteroid Field Visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    ax.legend(loc='lower left')

    ax.set_xlim(x_s - max_dist, x_s + max_dist)
    ax.set_ylim(y_s - max_dist, y_s + max_dist)

    plt.show()

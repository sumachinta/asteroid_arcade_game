import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple, List
from matplotlib.patches import Wedge, Patch
from matplotlib.lines import Line2D 

from utils.encoding import Ship, Asteroid

def visualize_game(
    ship: "Ship",
    asteroids: List["Asteroid"],
    d_threat: Optional[Tuple[float, float, float]] = None,
    max_dist: float = 400.0,
    theta_center_deg: float = 10.0,
):
    """
    Visualize the ship, asteroids, and optional directional threats.

    ship      : Ship instance
    asteroids : list of Asteroid instances
    d_threat  : optional (d_left, d_center, d_right)
    """

    x_s, y_s = ship.x, ship.y
    heading = ship.heading

    fig, ax = plt.subplots(figsize=(7, 7))

    # --- Ship triangle (larger + black) ---
    def ship_triangle(x, y, angle, scale=35):
        pts = np.array([[1, 0], [-0.5, 0.5], [-0.5, -0.5]]) * scale
        rot = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle),  np.cos(angle)]])
        pts = pts @ rot.T
        pts[:, 0] += x
        pts[:, 1] += y
        return pts

    tri = ship_triangle(x_s, y_s, heading)
    ax.fill(tri[:, 0], tri[:, 1], color="black", alpha=0.9, label="Ship")

    # Draw center sector as a shaded wedge 
    theta_c = np.radians(theta_center_deg)
    start_angle_deg = np.degrees(heading - theta_c)
    end_angle_deg   = np.degrees(heading + theta_c)

    wedge = Wedge(center=(x_s, y_s),r=max_dist,theta1=start_angle_deg,theta2=end_angle_deg,facecolor="lightgray",alpha=0.3,edgecolor=None)
    ax.add_patch(wedge)

    # Asteroids + dashed line toward ship 
    for ast in asteroids:
        # asteroid body
        ax.add_patch(
            plt.Circle((ast.x, ast.y), ast.size, alpha=0.6)
        )
        # line from asteroid to ship
        scale = 2.0  # tune for display
        ax.arrow(ast.x, ast.y,ast.vx * scale, ast.vy * scale,head_width=10, head_length=15,length_includes_head=True,color="red",alpha=0.7,)

    # Threat text 
    if d_threat is not None:
        d_left, d_center, d_right = d_threat
        text = (
            f"Threats:\n"
            f"Left   = {d_left:.2f}\n"
            f"Center = {d_center:.2f}\n"
            f"Right  = {d_right:.2f}"
        )
        ship_legend = Patch(facecolor='black', edgecolor='black', label='Ship')
        ax.text(
            0.02, 0.98, text,
            transform=ax.transAxes,
            fontsize=12,
            va='top',
            bbox=dict(facecolor='white', alpha=0.7)
        )
        heading_legend = Line2D([0], [0], linestyle='-', color='red', label='Heading direction')

    # --- Formatting ---
    ax.set_aspect('equal')
    ax.set_title("Asteroid Field Visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    ax.legend(handles = [ship_legend, heading_legend], loc='lower left')
        
    ax.set_xlim(x_s - max_dist, x_s + max_dist)
    ax.set_ylim(y_s - max_dist, y_s + max_dist)

    plt.show()

import math
import numpy as np
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Tuple

from utils.encoding import compute_directional_threat
from utils.encoding import WORLD_WIDTH, WORLD_HEIGHT, DT
from utils.encoding import Ship, Asteroid, Bullet, Threat, StimFreqs, GameState
from utils.encoding import  map_threat_to_stim_freqs
from utils.decoding import  Action, Heading, NeuralDecoder
from utils.spikes_simulate import FiringCounts, simulate_step_firing_counts
from utils.feedback import FeedbackState, FeedbackMode, FeedbackType, step_feedback, FEEDBACK_MODE_DEFAULT
from utils.game_physics import update_game_state



def make_initial_state() -> GameState:
    ship = Ship(
        x=WORLD_WIDTH / 2,
        y=WORLD_HEIGHT / 2,
        vx=0.0,
        vy=0.0,
        heading=0.0,
    )
    asteroids = [
        Asteroid(x=100.0, y=100.0, vx=20.0, vy=15.0, size=30.0),
    ]
    return GameState(ship=ship, asteroids=asteroids, bullets=[], t_s=0.0)


def run_simulation(num_steps: int, 
                   game_state: GameState = None,
                   feedback_mode_str: str = FEEDBACK_MODE_DEFAULT):
    """ Run the closed-loop simulation with optional feedback mode and initial game state.
    Prints out the state at each step."""
    
    feedback_mode = FeedbackMode[feedback_mode_str]
    state = make_initial_state() if game_state is None else game_state
    decoder = NeuralDecoder()
    fb_state = FeedbackState()
    sensory_pause_remaining = 0.0

    max_size = max(a.size for a in state.asteroids)

    for step in range(num_steps):
        print(f"\n=== STEP {step}, t={state.t_s:.3f} s ===")

        # ----- 1. Encoding: threat -> stim freqs (if not paused) -----
        if sensory_pause_remaining <= 0.0:
            threat = compute_directional_threat(ship=state.ship,asteroids=state.asteroids,max_size=max_size)
            stim_freqs = map_threat_to_stim_freqs(threat)
            print("Threat:", threat)
            print("Stim freqs:", stim_freqs)
        else:
            print(f"Sensory encoding PAUSED for {sensory_pause_remaining:.3f} s")

        # ----- 2. Simulated spikes for this 10 ms -----
        counts = simulate_step_firing_counts(bin_duration_s=DT)
        print("Firing counts:", counts)

        # ----- 3. Decode -> Action -----
        t_bin_end = state.t_s + DT
        action = decoder.step(counts, t_bin_end)
        print("Action:", action)

        # ----- 4. Physics + hit/kill detection -----
        hit, kill = update_game_state(state, action, DT)
        print("Hit:", hit, "Kill:", kill)
        print(
            f"Ship: x={state.ship.x:.1f}, y={state.ship.y:.1f}, "
            f"vx={state.ship.vx:.1f}, vy={state.ship.vy:.1f}, "
            f"heading_deg={math.degrees(state.ship.heading):.1f}"
        )

        # ----- 5. Feedback -----
        fb_type, pause_s, reset_game = step_feedback(
            feedback_mode, fb_state, hit, kill, DT
        )
        print("Feedback:", fb_type, "pause_sensory:", pause_s, "reset_game:", reset_game)

        # In real experiment, here you'd trigger reward/punishment stimulation
        # based on fb_type, we just manage the sensory pause + resets.

        if fb_type != FeedbackType.NONE:
            sensory_pause_remaining = pause_s

        if reset_game:
            print(">>> GAME RESET due to punishment")
            state = make_initial_state()
            max_size = max(a.size for a in state.asteroids)

        # decrease pause timer
        if sensory_pause_remaining > 0.0:
            sensory_pause_remaining = max(0.0, sensory_pause_remaining - DT)


# from copy import deepcopy
# from typing import Optional

# @dataclass
# class StepRecord:
#     t_s: float
#     ship: Ship
#     asteroids: List[Asteroid]
#     bullets: List[Bullet]
#     threat: Optional[Threat]
#     stim_freqs: Optional[StimFreqs]
#     counts: FiringCounts
#     action: Action
#     hit: bool
#     kill: bool
#     feedback_type: FeedbackType
#     sensory_pause_remaining: float


# def run_simulation_record(
#     num_steps: int = 200,
#     feedback_mode_str: str = FEEDBACK_MODE_DEFAULT,
# ) -> List[StepRecord]:
#     """
#     Run the closed-loop simulation and return a list of StepRecord objects,
#     one per 10 ms step, for visualization.
#     """
#     feedback_mode = FeedbackMode[feedback_mode_str]
#     state = make_initial_state()
#     decoder = NeuralDecoder(bin_duration_s=DT)
#     fb_state = FeedbackState()
#     sensory_pause_remaining = 0.0

#     max_size = max(a.size for a in state.asteroids)

#     history: List[StepRecord] = []

#     for step in range(num_steps):

#         # 1. Encoding (if not paused)
#         if sensory_pause_remaining <= 0.0:
#             threat = compute_directional_threat(
#                 ship=state.ship,
#                 asteroids=state.asteroids,
#                 max_size=max_size,
#             )
#             stim_freqs = map_threat_to_stim_freqs(threat)
#         else:
#             threat = None
        #     stim_freqs = None

        # # 2. Simulated spikes
        # counts = simulate_step_firing_counts(bin_duration_s=DT)

        # # 3. Decode -> action
        # t_bin_end = state.t_s + DT
        # action = decoder.step(counts, t_bin_end)

        # # 4. Physics + events
        # hit, kill = update_game_state(state, action, DT)

        # # 5. Feedback
        # fb_type, pause_s, reset_game = step_feedback(
        #     feedback_mode, fb_state, hit, kill, DT
        # )

        # if fb_type != FeedbackType.NONE:
        #     sensory_pause_remaining = pause_s

        # if reset_game:
        #     state = make_initial_state()
        #     max_size = max(a.size for a in state.asteroids)

        # # Decrease remaining pause time
        # if sensory_pause_remaining > 0.0:
        #     sensory_pause_remaining = max(
        #         0.0, sensory_pause_remaining - DT
        #     )

    #     # 6. Store a snapshot (deepcopy to freeze state)
    #     record = StepRecord(
    #         t_s=state.t_s,
    #         ship=deepcopy(state.ship),
    #         asteroids=deepcopy(state.asteroids),
    #         bullets=deepcopy(state.bullets),
    #         threat=deepcopy(threat),
    #         stim_freqs=deepcopy(stim_freqs),
    #         counts=counts,
    #         action=action,
    #         hit=hit,
    #         kill=kill,
    #         feedback_type=fb_type,
    #         sensory_pause_remaining=sensory_pause_remaining,
    #     )
    #     history.append(record)

    # return history


# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# def _draw_ship(ax, ship: Ship):
#     """Draw the ship as a small oriented triangle."""
#     # Triangle in local coords
#     r = 12.0
#     back = 6.0
#     pts = np.array([
#         [ r, 0.0],   # nose
#         [-back, 5.0],
#         [-back,-5.0],
#     ])
#     # Rotate & translate
#     c = math.cos(ship.heading)
#     s = math.sin(ship.heading)
#     R = np.array([[c, -s],
#                   [s,  c]])
#     pts_world = pts @ R.T + np.array([ship.x, ship.y])
#     ax.fill(pts_world[:,0], pts_world[:,1], edgecolor="black", facecolor="none", linewidth=1.0)


# def animate_history(history: List[StepRecord],
#                     save_path: str | None = None,
#                     fps: int = 30):
#     """
#     Visualize the whole simulation as an animation.
#     If save_path is provided (e.g. 'run.mp4'), saves a video.
#     Requires ffmpeg installed to save mp4.
#     """
#     fig, ax = plt.subplots(figsize=(6, 4))
#     ax.set_xlim(0, WORLD_WIDTH)
#     ax.set_ylim(0, WORLD_HEIGHT)
#     ax.set_aspect("equal")
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")

#     ship_patch = None
#     asteroid_patches = []
#     bullet_scatter = ax.scatter([], [], s=10, c="black")
#     title = ax.set_title("")

#     def init():
#         nonlocal ship_patch, asteroid_patches
#         ship_patch = None
#         asteroid_patches = []
#         bullet_scatter.set_offsets(np.empty((0, 2)))
#         ax.set_xlim(0, WORLD_WIDTH)
#         ax.set_ylim(0, WORLD_HEIGHT)
#         return []

#     def update(frame_idx):
#         nonlocal ship_patch, asteroid_patches
#         rec = history[frame_idx]
#         ax.clear()
#         ax.set_xlim(0, WORLD_WIDTH)
#         ax.set_ylim(0, WORLD_HEIGHT)
#         ax.set_aspect("equal")
#         ax.set_xlabel("X")
#         ax.set_ylabel("Y")

#         # Background color hints sensory pause
#         if rec.sensory_pause_remaining > 0:
#             ax.set_facecolor("#ffeeee")
#         else:
#             ax.set_facecolor("white")

#         # Draw ship
#         _draw_ship(ax, rec.ship)

#         # Ship color overlay for hit/kill
#         if rec.hit:
#             ax.text(rec.ship.x, rec.ship.y, "X", color="red", fontsize=14,
#                     ha="center", va="center", fontweight="bold")
#         elif rec.kill:
#             ax.text(rec.ship.x, rec.ship.y, "✓", color="green", fontsize=14,
#                     ha="center", va="center", fontweight="bold")

#         # Asteroids
#         for a in rec.asteroids:
#             circ = plt.Circle((a.x, a.y), radius=a.size,
#                               edgecolor="gray", facecolor="none", linewidth=1.0)
#             ax.add_patch(circ)

#         # Bullets
#         if rec.bullets:
#             xs = [b.x for b in rec.bullets if b.alive]
#             ys = [b.y for b in rec.bullets if b.alive]
#             ax.scatter(xs, ys, s=8, c="black")

#         # Title: time + feedback + threat (if available)
#         fb = rec.feedback_type.name
#         th_text = ""
#         if rec.threat is not None:
#             th_text = f" | Threat L/C/R = {rec.threat.left:.2f}/{rec.threat.center:.2f}/{rec.threat.right:.2f}"
#         ax.set_title(f"t={rec.t_s:.3f}s | fb={fb}{th_text}")

#         return []

#     anim = FuncAnimation(
#         fig,
#         update,
#         frames=len(history),
#         init_func=init,
#         interval=1000 * DT,  # ms per frame
#         blit=False,
#     )

#     if save_path is not None:
#         print(f"Saving animation to {save_path} ...")
#         anim.save(save_path, fps=fps)
#         print("Done.")

#     plt.show()


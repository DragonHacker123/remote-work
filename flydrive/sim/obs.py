"""The observation vector -- the contract between the simulator and F1 25.

Every field here is recoverable from F1 25's own UDP telemetry given a
centreline recorded once per circuit, which is the whole point: a policy
trained in the simulator sees the *same* vector in the game, so transfer is a
vehicle-dynamics problem rather than a representation problem.

Where each field comes from in the game:

    vx, vy, yaw_rate      Motion / Motion Ex (packet 0 / 13) world velocity
                          rotated into the car frame; yaw rate differentiated
                          from the Motion packet's yaw.
    e_y, e_psi, pursuit_* world position and yaw vs the recorded centreline.
    kappa_*               precomputed from the centreline.
    last_*                the commands we sent on the previous tick.
    slip_front/rear       Motion Ex wheel slip angles, front/rear averaged.
"""

from __future__ import annotations

import numpy as np

# Lookahead ladder for the curvature preview. The far end has to cover a full
# braking zone -- roughly 120 m from top speed down to a slow corner -- or a
# controller literally cannot see the corner in time to stop for it.
PREVIEW_DISTANCES = np.array([8.0, 16.0, 28.0, 45.0, 65.0, 90.0, 120.0, 155.0, 195.0])
# Aim points for pure pursuit. A fixed lookahead cannot work across a 20-90
# m/s speed range: too far and the car cuts every apex, too near and the
# steering loop oscillates. Three samples let the controller interpolate an
# aim point at roughly a constant *time* ahead.
PURSUIT_DISTANCES = np.array([12.0, 40.0, 90.0])
PURSUIT_TIME = 1.1        # seconds of lookahead the driver targets

OBS_NAMES: list[str] = (
    ["vx", "vy", "yaw_rate", "e_y", "e_psi_sin", "e_psi_cos"]
    + [f"pursuit_{int(d)}" for d in PURSUIT_DISTANCES]
    + [f"kappa_{int(d)}" for d in PREVIEW_DISTANCES]
    + ["last_steer", "last_throttle", "last_brake", "slip_front", "slip_rear"]
    + ["yaw_sin", "yaw_cos", "goal_sin", "goal_cos"]
)

OBS_DIM = len(OBS_NAMES)

# Derived index constants, so nothing downstream hard-codes a column number.
IDX_VX = OBS_NAMES.index("vx")
IDX_VY = OBS_NAMES.index("vy")
IDX_YAW_RATE = OBS_NAMES.index("yaw_rate")
IDX_EY = OBS_NAMES.index("e_y")
IDX_EPSI_SIN = OBS_NAMES.index("e_psi_sin")
IDX_EPSI_COS = OBS_NAMES.index("e_psi_cos")
PURSUIT_SLICE = slice(
    OBS_NAMES.index(f"pursuit_{int(PURSUIT_DISTANCES[0])}"),
    OBS_NAMES.index(f"pursuit_{int(PURSUIT_DISTANCES[-1])}") + 1,
)
KAPPA_SLICE = slice(
    OBS_NAMES.index(f"kappa_{int(PREVIEW_DISTANCES[0])}"),
    OBS_NAMES.index(f"kappa_{int(PREVIEW_DISTANCES[-1])}") + 1,
)

# The last four entries are absolute world angles rather than scale-free
# features. They are consumed only by the central-complex ring encoder, which
# needs a heading and a goal *direction*, not a precomputed error -- computing
# that error is the circuit's job and we must not do it for it.
RING_SLICE = slice(OBS_DIM - 4, OBS_DIM)
FEATURE_SLICE = slice(0, OBS_DIM - 4)

# Normalisation constants, shared by simulator and bridge.
V_SCALE = 80.0
VY_SCALE = 8.0
R_SCALE = 1.5
KAPPA_SCALE = 0.04
SLIP_SCALE = 0.25


def make_obs(
    vx: np.ndarray,
    vy: np.ndarray,
    yaw_rate: np.ndarray,
    e_y: np.ndarray,
    e_psi: np.ndarray,
    pursuit: np.ndarray,
    kappa_preview: np.ndarray,
    last_action: np.ndarray,
    slip_front: np.ndarray,
    slip_rear: np.ndarray,
    yaw: np.ndarray,
    goal_angle: np.ndarray,
    half_width: float,
) -> np.ndarray:
    """Assemble the (B, OBS_DIM) observation."""
    return np.column_stack(
        [
            vx / V_SCALE,
            vy / VY_SCALE,
            yaw_rate / R_SCALE,
            e_y / half_width,
            np.sin(e_psi),
            np.cos(e_psi),
            pursuit / (np.pi / 2),
            kappa_preview / KAPPA_SCALE,
            last_action,
            np.clip(slip_front / SLIP_SCALE, -3.0, 3.0),
            np.clip(slip_rear / SLIP_SCALE, -3.0, 3.0),
            np.sin(yaw),
            np.cos(yaw),
            np.sin(goal_angle),
            np.cos(goal_angle),
        ]
    )


def slip_angles(vx, vy, yaw_rate, steer_cmd, params):
    """Front and rear slip angles from the bicycle-model relations.

    Identical to what the simulator's tyre model computes internally, so the
    observation means the same thing whether it came from the simulator or from
    the game.
    """
    delta = np.clip(steer_cmd, -1.0, 1.0) * params.max_steer
    vx_safe = np.maximum(vx, 5.0)
    return (
        np.arctan2(vy + params.lf * yaw_rate, vx_safe) - delta,
        np.arctan2(vy - params.lr * yaw_rate, vx_safe),
    )


def observe(
    track,
    x: np.ndarray,
    y: np.ndarray,
    heading: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    yaw_rate: np.ndarray,
    last_action: np.ndarray,
    hint: np.ndarray,
    params,
    slip: tuple[np.ndarray, np.ndarray] | None = None,
):
    """Build the observation from raw car state against a track.

    This is the single implementation shared by the simulator and the F1 25
    bridge. Keeping it in one place is what makes "the network sees the same
    vector in the game as in training" a fact rather than an aspiration.

    Returns ``(obs, idx, s, e_y, e_psi, goal_angle)``.
    """
    idx, s, e_y, psi_ref = track.project(x, y, hint)
    e_psi = (heading - psi_ref + np.pi) % (2.0 * np.pi) - np.pi

    aims = np.stack(
        [
            np.arctan2(p[:, 1] - y, p[:, 0] - x)
            for p in (track.point_at(idx, d) for d in PURSUIT_DISTANCES)
        ],
        axis=1,
    )
    want = np.clip(PURSUIT_TIME * vx, PURSUIT_DISTANCES[0], PURSUIT_DISTANCES[-1])
    step = np.round(want / track.ds).astype(np.int64)
    gp = track.xy[(idx + step) % track.n]
    goal_angle = np.arctan2(gp[:, 1] - y, gp[:, 0] - x)

    if slip is None:
        slip = slip_angles(vx, vy, yaw_rate, last_action[:, 0], params)

    obs = make_obs(
        vx=vx,
        vy=vy,
        yaw_rate=yaw_rate,
        e_y=e_y,
        e_psi=e_psi,
        pursuit=(aims - heading[:, None] + np.pi) % (2.0 * np.pi) - np.pi,
        kappa_preview=track.preview(idx, PREVIEW_DISTANCES),
        last_action=last_action,
        slip_front=slip[0],
        slip_rear=slip[1],
        yaw=heading,
        goal_angle=goal_angle,
        half_width=track.half_width,
    )
    return obs, idx, s, e_y, e_psi, goal_angle


def pursuit_angle(obs: np.ndarray) -> np.ndarray:
    """Aim-point angle at a roughly constant time ahead, from the three samples.

    Interpolates between the fixed lookahead distances so the effective aim
    point sits ``PURSUIT_TIME`` seconds down the road at any speed.
    """
    samples = obs[:, PURSUIT_SLICE] * (np.pi / 2)
    vx = obs[:, IDX_VX] * V_SCALE
    want = np.clip(PURSUIT_TIME * vx, PURSUIT_DISTANCES[0], PURSUIT_DISTANCES[-1])
    return np.array(
        [np.interp(w, PURSUIT_DISTANCES, row) for w, row in zip(want, samples)]
    )


def ring_angles(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover (heading, goal) absolute angles from an observation batch."""
    ys, yc, gs, gc = (obs[:, i] for i in range(RING_SLICE.start, OBS_DIM))
    return np.arctan2(ys, yc), np.arctan2(gs, gc)

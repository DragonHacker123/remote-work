"""A classical reference driver: Stanley feedback plus a friction-limited speed profile.

This is the yardstick. It is deterministic, has no training variance, and is
built from textbook vehicle dynamics rather than learning, so it answers "how
fast *should* a lap be?" before any brain is involved. The connectome network
is judged against it.

It is a reference, not an optimum: it tracks the centreline rather than a
minimum-curvature racing line, so a good learned policy can and should beat it
by a few percent by straightening the corners out.

Notably it reads *only* the shared observation vector, so the exact same driver
runs in F1 25 through the bridge.
"""

from __future__ import annotations

import numpy as np

from ..sim.obs import (
    IDX_EPSI_COS,
    IDX_EPSI_SIN,
    IDX_EY,
    IDX_VX,
    IDX_YAW_RATE,
    KAPPA_SCALE,
    KAPPA_SLICE,
    PREVIEW_DISTANCES,
    R_SCALE,
    V_SCALE,
)
from ..sim.vehicle import G, VehicleParams


def corner_speed(kappa: np.ndarray, p: VehicleParams, v_top: float = 95.0) -> np.ndarray:
    """Steady-state speed a corner of curvature ``kappa`` supports.

    Solving ``v^2 kappa = mu (m g + k_down v^2) / m`` for v gives a closed
    form. When downforce alone covers the requirement the corner is flat and
    the limit is top speed.
    """
    k = np.abs(kappa)
    denom = k * p.mass - p.mu * p.k_down
    with np.errstate(divide="ignore", invalid="ignore"):
        v2 = np.where(denom > 1e-6, p.mu * p.mass * G / np.maximum(denom, 1e-6), np.inf)
    return np.minimum(np.sqrt(np.maximum(v2, 0.0)), v_top)


class ReferenceDriver:
    """Curvature feedforward + Stanley feedback, with a backward speed profile.

    Pure pursuit was tried first and abandoned. A fixed aim point cannot work
    across a 20-90 m/s speed range -- far enough to be stable at 90 cuts every
    apex at 30, and near enough to be precise at 30 sends the steering loop
    into oscillation at 90. Stanley's cross-track term is divided by speed,
    which gives the required behaviour at both ends for free.
    """

    def __init__(
        self,
        params: VehicleParams | None = None,
        k_heading: float = 0.85,
        k_cross: float = 1.6,
        k_damp: float = 0.06,
        k_throttle: float = 0.18,
        k_brake: float = 0.11,
        speed_margin: float = 0.90,
        brake_safety: float = 0.85,
        v_top: float = 92.0,
        half_width: float = 6.0,
    ):
        self.p = params or VehicleParams()
        self.brake_safety = brake_safety
        self.k_heading = k_heading
        self.k_cross = k_cross
        self.k_damp = k_damp
        self.half_width = half_width
        self.k_throttle = k_throttle
        self.k_brake = k_brake
        self.speed_margin = speed_margin
        self.v_top = v_top

        # Fixed linear-interpolation matrix from the sparse curvature preview
        # onto a 4 m grid, so the backward pass is one matmul plus a loop over
        # grid points rather than a per-row interpolation.
        self._fine = np.arange(PREVIEW_DISTANCES[0], PREVIEW_DISTANCES[-1] + 1e-9, 4.0)
        k = len(PREVIEW_DISTANCES)
        weights = np.zeros((len(self._fine), k))
        for i, d in enumerate(self._fine):
            j = int(np.clip(np.searchsorted(PREVIEW_DISTANCES, d), 1, k - 1))
            d0, d1 = PREVIEW_DISTANCES[j - 1], PREVIEW_DISTANCES[j]
            t = (d - d0) / (d1 - d0)
            weights[i, j - 1] = 1.0 - t
            weights[i, j] = t
        self._interp = weights

    def reset(self, n: int) -> None:  # stateless, but keeps the policy protocol
        pass

    def target_speed(self, vx: np.ndarray, kappa_preview: np.ndarray) -> np.ndarray:
        """Fastest speed now that still allows braking for every previewed corner.

        Checking each preview sample independently is not enough: it misses
        corners that fall between samples, and it evaluates downforce at entry
        speed, which flatters the available deceleration. Instead the sparse
        preview is resampled onto a fine grid and the speed profile integrated
        backwards from the far end, recomputing grip at each step -- the same
        backward pass a lap-time simulator uses.
        """
        p = self.p
        kappa_fine = np.abs(kappa_preview) @ self._interp.T     # (B, M)
        v_limit = corner_speed(kappa_fine, p, self.v_top)
        step = self._fine[1] - self._fine[0]

        v = v_limit[:, -1]
        for j in range(v_limit.shape[1] - 2, -1, -1):
            a_tyre = p.mu * (p.mass * G + p.k_down * v * v) / p.mass
            a = np.minimum(a_tyre, p.brake_force / p.mass) * self.brake_safety
            a = a + p.k_drag * v * v / p.mass
            v = np.minimum(v_limit[:, j], np.sqrt(v * v + 2.0 * a * step))
        return self.speed_margin * np.minimum(v, self.v_top)

    @property
    def understeer_gradient(self) -> float:
        """K_us in rad per m/s^2: extra steering needed per unit of lateral g."""
        p = self.p
        wheelbase = p.lf + p.lr
        front_load = p.mass * G * p.lr / wheelbase
        rear_load = p.mass * G * p.lf / wheelbase
        return front_load / p.c_alpha_f - rear_load / p.c_alpha_r

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        p = self.p
        vx = obs[:, IDX_VX] * V_SCALE
        yaw_rate = obs[:, IDX_YAW_RATE] * R_SCALE
        e_y = obs[:, IDX_EY] * self.half_width                       # metres
        e_psi = np.arctan2(obs[:, IDX_EPSI_SIN], obs[:, IDX_EPSI_COS])
        kappa = obs[:, KAPPA_SLICE] * KAPPA_SCALE

        # Feedforward: the steering angle the corner actually needs, Ackermann
        # plus the understeer correction. Without this the feedback terms have
        # to generate the whole steering input from error, which means the car
        # is always lagging the corner.
        kappa_now = kappa[:, 0]
        wheelbase = p.lf + p.lr
        delta_ff = wheelbase * kappa_now + self.understeer_gradient * vx * vx * kappa_now

        # Stanley feedback. The cross-track term is divided by speed, which is
        # what keeps the loop stable at 90 m/s without making it sluggish at
        # 30 -- a fixed-lookahead pure pursuit cannot do both.
        delta_fb = (
            -self.k_heading * e_psi
            - np.arctan2(self.k_cross * e_y, np.maximum(vx, 5.0))
            - self.k_damp * (yaw_rate - vx * kappa_now)
        )
        steer = np.clip((delta_ff + delta_fb) / p.max_steer, -1.0, 1.0)

        v_target = self.target_speed(vx, kappa)
        err = v_target - vx
        throttle = np.clip(self.k_throttle * err, 0.0, 1.0)
        brake = np.clip(-self.k_brake * err, 0.0, 1.0)

        # Friction ellipse, applied by the driver rather than discovered by
        # crashing: grip already spent on cornering is not available for
        # acceleration. Flooring it mid-corner is exactly how you spin.
        a_lat = np.abs(vx * yaw_rate)
        a_max = p.mu * (p.mass * G + p.k_down * vx * vx) / p.mass
        lat_frac = np.clip(a_lat / np.maximum(a_max, 1.0), 0.0, 0.98)
        long_avail = np.sqrt(1.0 - lat_frac * lat_frac)
        throttle = np.minimum(throttle, long_avail)
        brake = np.minimum(brake, long_avail)
        return np.stack([steer, throttle, brake], axis=1)

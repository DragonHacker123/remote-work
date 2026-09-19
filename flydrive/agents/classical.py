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
    GRADE_DISTANCES,
    GRADE_SCALE,
    GRADE_SLICE,
    KAPPA_SCALE,
    KAPPA_SLICE,
    PREVIEW_DISTANCES,
    R_SCALE,
    V_SCALE,
    VCURV_SCALE,
    VCURV_SLICE,
)
from ..sim.vehicle import G, VehicleParams


def corner_speed(
    kappa: np.ndarray, p: VehicleParams, v_top: float = 95.0,
    load: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Steady-state speed a corner of curvature ``kappa`` supports.

    Solving ``v^2 kappa = mu (m g + k_down v^2) / m`` for v gives a closed
    form. When downforce alone covers the requirement the corner is flat and
    the limit is top speed.
    """
    k = np.abs(kappa)
    denom = k * p.mass - p.mu * p.k_down
    with np.errstate(divide="ignore", invalid="ignore"):
        v2 = np.where(
            denom > 1e-6, p.mu * p.mass * G * load / np.maximum(denom, 1e-6), np.inf
        )
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

        # Same idea for the gradient preview, which is sampled more coarsely.
        gw = np.zeros((len(self._fine), len(GRADE_DISTANCES)))
        gk = len(GRADE_DISTANCES)
        for i, d in enumerate(self._fine):
            j = int(np.clip(np.searchsorted(GRADE_DISTANCES, d), 1, gk - 1))
            d0, d1 = GRADE_DISTANCES[j - 1], GRADE_DISTANCES[j]
            t = np.clip((d - d0) / (d1 - d0), 0.0, 1.0)
            gw[i, j - 1] = 1.0 - t
            gw[i, j] = t
        self._grade_interp = gw

    def reset(self, n: int) -> None:  # stateless, but keeps the policy protocol
        pass

    def target_speed(
        self,
        vx: np.ndarray,
        kappa_preview: np.ndarray,
        grade_preview: np.ndarray | None = None,
        vcurv_preview: np.ndarray | None = None,
    ) -> np.ndarray:
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
        step = self._fine[1] - self._fine[0]

        # Gravity along the road changes what braking is available: downhill
        # into a corner is the classic way to arrive too fast. Vertical
        # curvature changes how much grip there is at all -- Les Combes sits
        # over a crest at the top of the Kemmel climb, so the car goes light
        # exactly where it is braking hardest.
        zeros = np.zeros_like(kappa_fine)
        grade_fine = zeros if grade_preview is None else grade_preview @ self._grade_interp.T
        vcurv_fine = zeros if vcurv_preview is None else vcurv_preview @ self._grade_interp.T
        cos_theta = 1.0 / np.sqrt(1.0 + grade_fine * grade_fine)

        v = corner_speed(kappa_fine[:, -1], p, self.v_top)
        for j in range(kappa_fine.shape[1] - 2, -1, -1):
            load = np.clip(cos_theta[:, j] + v * v * vcurv_fine[:, j] / G, 0.15, 2.5)
            limit = corner_speed(kappa_fine[:, j], p, self.v_top, load)
            a_tyre = p.mu * (p.mass * G * load + p.k_down * v * v) / p.mass
            a = np.minimum(a_tyre, p.brake_force / p.mass) * self.brake_safety
            a = a + p.k_drag * v * v / p.mass + G * grade_fine[:, j]
            v = np.minimum(limit, np.sqrt(v * v + 2.0 * np.maximum(a, 0.5) * step))
        return self.speed_margin * np.minimum(v, self.v_top)

    def rear_brake_limit(
        self, vx: np.ndarray, a_lat: np.ndarray, load: np.ndarray
    ) -> np.ndarray:
        """Brake force the rear axle can take while cornering at ``a_lat``.

        The whole-car friction ellipse is not the binding constraint on corner
        entry: the rear axle is. It carries the smaller share of the weight,
        braking transfers load off it, and the moment it saturates the car
        rotates. Because the load depends on the very force being solved for,
        the ellipse is a quadratic in brake force rather than a simple cap.

        Lateral force is split between the axles in inverse proportion to their
        distance from the centre of mass, which is the zero-yaw-moment
        condition.
        """
        p = self.p
        wheelbase = p.lf + p.lr
        rest = (
            p.mass * G * load * p.lf / wheelbase
            + (1.0 - p.aero_balance) * p.k_down * vx * vx
        )
        fy = p.mass * a_lat * p.lf / wheelbase
        c = p.h_cog / wheelbase
        share = 1.0 - p.brake_bias
        a = max(share * share - (p.mu * c) ** 2, 1e-3)
        b = 2.0 * p.mu * p.mu * rest * c
        const = fy * fy - (p.mu * rest) ** 2
        disc = np.maximum(b * b - 4.0 * a * const, 0.0)
        return np.maximum((-b + np.sqrt(disc)) / (2.0 * a), 0.0)

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
        grade = obs[:, GRADE_SLICE] * GRADE_SCALE
        vcurv = obs[:, VCURV_SLICE] * VCURV_SCALE

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

        v_target = self.target_speed(vx, kappa, grade, vcurv)
        err = v_target - vx
        throttle = np.clip(self.k_throttle * err, 0.0, 1.0)
        brake = np.clip(-self.k_brake * err, 0.0, 1.0)

        # Friction ellipse, applied by the driver rather than discovered by
        # crashing: grip already spent on cornering is not available for
        # acceleration. Flooring it mid-corner is exactly how you spin.
        #
        # The first version of this compared *pedal travel* against a *force*
        # fraction. Full brake is 32 kN, roughly twice what the tyres can take
        # at 35 m/s, so "brake <= 0.78" still permitted a demand well past the
        # limit -- and it checked the car as a whole, when what actually lets go
        # on corner entry is the rear axle by itself. On Spa's approach to La
        # Source that combination retired the reference driver every lap.
        slope2 = 1.0 + grade[:, 0] * grade[:, 0]
        load = np.clip(1.0 / np.sqrt(slope2) + vx * vx * vcurv[:, 0] / G, 0.15, 2.5)
        grip = p.mu * (p.mass * G * load + p.k_down * vx * vx)        # newtons
        a_lat = np.abs(vx * yaw_rate)
        lat_frac = np.clip(p.mass * a_lat / np.maximum(grip, 1.0), 0.0, 0.98)
        long_force = grip * np.sqrt(1.0 - lat_frac * lat_frac)

        drive_cap = np.minimum(p.power / np.maximum(vx, 8.0), grip)
        throttle = np.minimum(throttle, long_force / np.maximum(drive_cap, 1.0))
        brake = np.minimum(
            brake,
            np.minimum(long_force, self.rear_brake_limit(vx, a_lat, load)) / p.brake_force,
        )
        return np.stack([steer, throttle, brake], axis=1)

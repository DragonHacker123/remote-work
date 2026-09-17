"""Dynamic bicycle model with load-sensitive tyres and aerodynamic downforce.

Enough vehicle dynamics that the car can be driven over the limit: it
understeers, it snaps into oversteer if you ask for lateral grip you have
already spent longitudinally, and its cornering speed rises with downforce the
way a real single-seater's does. Those are the behaviours a driving policy has
to learn, so leaving them out would make the task the wrong shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G = 9.81


@dataclass(frozen=True)
class VehicleParams:
    """Roughly current-generation F1: 798 kg, ~1000 hp, big aero."""

    mass: float = 798.0
    inertia: float = 1100.0
    lf: float = 1.70              # CoG to front axle, m
    lr: float = 1.90              # CoG to rear axle, m
    mu: float = 1.50              # peak tyre friction coefficient
    c_alpha_f: float = 180_000.0  # front cornering stiffness, N/rad
    c_alpha_r: float = 200_000.0
    shape_c: float = 1.35         # Pacejka shape factor
    # Downforce ~3.7x car weight at top speed, drag giving ~325 km/h. Push
    # k_down much higher and every corner becomes flat-out at exactly 100% of
    # the friction limit, which leaves the controller no margin at all.
    k_down: float = 3.60          # downforce coefficient, N/(m/s)^2
    aero_balance: float = 0.45    # fraction of downforce on the front axle
    k_drag: float = 1.05          # drag coefficient, N/(m/s)^2
    power: float = 735_000.0      # W
    brake_force: float = 32_000.0 # N at the tyre contact patch
    brake_bias: float = 0.60      # fraction of braking taken by the front axle
    h_cog: float = 0.30           # CoG height, m -- sets longitudinal load transfer
    max_steer: float = 0.30       # rad at the roadwheel
    steer_rate: float = 4.0       # rad/s, limits how fast steering can change
    rolling: float = 300.0        # N


def _rear_load_before_transfer(p: VehicleParams, speed2: np.ndarray) -> np.ndarray:
    """Rear axle load ignoring longitudinal transfer.

    Used only to cap traction-limited drive force, which would otherwise be a
    circular dependency: drive sets the transfer that sets the load that caps
    the drive.
    """
    static_r = p.mass * G * p.lf / (p.lf + p.lr)
    return static_r + (1.0 - p.aero_balance) * p.k_down * speed2


def _magic(alpha: np.ndarray, peak: np.ndarray, c_alpha: float, shape_c: float) -> np.ndarray:
    """Simplified Pacejka: F = D sin(C arctan(B alpha)).

    ``B`` is set from the cornering stiffness so the curve has the right slope
    at zero slip and saturates at ``peak``.
    """
    peak = np.maximum(peak, 1.0)
    b = c_alpha / (shape_c * peak)
    return -peak * np.sin(shape_c * np.arctan(b * alpha))


def step_dynamics(
    state: np.ndarray,
    steer: np.ndarray,
    throttle: np.ndarray,
    brake: np.ndarray,
    p: VehicleParams,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advance ``state`` (B, 6) = [X, Y, psi, vx, vy, r] by ``dt``.

    Returns ``(new_state, alpha_f, alpha_r)``.
    """
    x, y, psi, vx, vy, r = (state[:, i] for i in range(6))
    delta = np.clip(steer, -1.0, 1.0) * p.max_steer

    # Slip angles. Below a few m/s these are ill-conditioned, so the
    # denominator is floored -- the usual kinematic/dynamic blend in disguise.
    vx_safe = np.maximum(vx, 5.0)
    alpha_f = np.arctan2(vy + p.lf * r, vx_safe) - delta
    alpha_r = np.arctan2(vy - p.lr * r, vx_safe)

    speed2 = vx * vx + vy * vy
    downforce = p.k_down * speed2
    wheelbase = p.lf + p.lr
    static_f = p.mass * G * p.lr / wheelbase
    static_r = p.mass * G * p.lf / wheelbase

    # Longitudinal force generated *at the tyres*: power-limited above ~30 m/s,
    # traction-limited below.
    drive = throttle * np.minimum(
        p.power / np.maximum(vx, 8.0), p.mu * _rear_load_before_transfer(p, speed2)
    )
    # Braking is limited by grip as well as by the brakes. Without this the
    # pedal can demand more force than all four tyres can transmit, which in a
    # model with no wheel-lock dynamics just deletes the car's lateral grip.
    grip_limit = p.mu * (p.mass * G + p.k_down * speed2)
    stop = np.minimum(brake * p.brake_force, grip_limit)
    f_long = drive - stop
    drag = p.k_drag * speed2 + p.rolling * np.sign(vx)
    fx = f_long - drag

    # Longitudinal load transfer. Braking loads the front, which is precisely
    # why a car can brake and turn at the same time.
    transfer = f_long * p.h_cog / wheelbase
    fz_f = np.maximum(static_f + p.aero_balance * downforce - transfer, 100.0)
    fz_r = np.maximum(static_r + (1.0 - p.aero_balance) * downforce + transfer, 100.0)

    # Friction ellipse, per axle. Two things must NOT be charged to the tyres
    # here: aerodynamic drag, which acts on the body rather than the contact
    # patch, and the front share of braking, which the rear axle never sees.
    # Charging either to the rear costs it most of its lateral grip and turns
    # every corner entry into a spin.
    cap_f = p.mu * fz_f
    cap_r = p.mu * fz_r
    braking = np.maximum(-f_long, 0.0)
    driving = np.maximum(f_long, 0.0)
    long_f = p.brake_bias * braking
    long_r = (1.0 - p.brake_bias) * braking + driving

    used_f = np.clip(long_f / np.maximum(cap_f, 1.0), 0.0, 0.98)
    used_r = np.clip(long_r / np.maximum(cap_r, 1.0), 0.0, 0.98)
    peak_f = cap_f * np.sqrt(1.0 - used_f * used_f)
    peak_r = cap_r * np.sqrt(1.0 - used_r * used_r)

    fy_f = _magic(alpha_f, peak_f, p.c_alpha_f, p.shape_c)
    fy_r = _magic(alpha_r, peak_r, p.c_alpha_r, p.shape_c)

    ax = (fx - fy_f * np.sin(delta)) / p.mass + vy * r
    ay = (fy_f * np.cos(delta) + fy_r) / p.mass - vx * r
    dr = (p.lf * fy_f * np.cos(delta) - p.lr * fy_r) / p.inertia

    vx_n = np.maximum(vx + ax * dt, 0.0)
    vy_n = vy + ay * dt
    r_n = r + dr * dt
    psi_n = psi + r_n * dt
    x_n = x + (vx_n * np.cos(psi) - vy_n * np.sin(psi)) * dt
    y_n = y + (vx_n * np.sin(psi) + vy_n * np.cos(psi)) * dt

    return np.stack([x_n, y_n, psi_n, vx_n, vy_n, r_n], axis=1), alpha_f, alpha_r


def cornering_limit(speed: np.ndarray, p: VehicleParams) -> np.ndarray:
    """Steady-state lateral acceleration available at a given speed, m/s^2.

    Used by the reference driver to build a friction-limited speed profile.
    """
    downforce = p.k_down * speed * speed
    return p.mu * (p.mass * G + downforce) / p.mass

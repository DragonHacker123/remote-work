import numpy as np
import pytest

from flydrive.agents.classical import ReferenceDriver, corner_speed
from flydrive.sim import get_track, make_env
from flydrive.sim.track import TRACKS, from_layout
from flydrive.sim.vehicle import G, VehicleParams, step_dynamics


# --------------------------------------------------------------------- track


@pytest.mark.parametrize("name", sorted(TRACKS))
def test_tracks_close_on_themselves(name):
    track = get_track(name)
    gap = np.linalg.norm(track.xy[-1] - track.xy[0])
    assert gap < 0.01 * track.length, f"{name} leaves a {gap:.1f} m gap"


def test_curvature_of_a_circle_matches_one_over_radius():
    radius = 80.0
    track = from_layout([("corner", radius, 360)], ds=0.5)
    # Ignore the smoothed transition at the seam.
    middle = track.kappa[len(track.kappa) // 4 : 3 * len(track.kappa) // 4]
    assert np.allclose(middle, 1.0 / radius, rtol=0.08)


def test_projection_recovers_a_known_lateral_offset():
    track = get_track("oval")
    i = 400
    offset = 3.0
    psi = track.psi[i]
    # Positive e_y is to the left of the direction of travel.
    x = np.array([track.xy[i, 0] - np.sin(psi) * offset])
    y = np.array([track.xy[i, 1] + np.cos(psi) * offset])
    _idx, _s, e_y, _ref = track.project(x, y, np.array([i]))
    assert e_y[0] == pytest.approx(offset, abs=0.05)


def test_layout_produces_a_spread_of_corner_speeds():
    """A circuit that is flat out everywhere is not a driving task."""
    track = get_track("national")
    v = np.minimum(corner_speed(track.kappa, VehicleParams()), 92.0)
    assert v.min() < 35.0, "no slow corners"
    assert np.mean(v >= 91.0) < 0.9, "almost everything is flat out"


# ------------------------------------------------------------------- vehicle


def peak_lateral_force(speed, throttle=0.0, brake=0.0, p=None):
    """Largest lateral force the tyres produce, swept over slip angle.

    Swept in slip angle rather than lateral velocity: the tyre peak sits at a
    roughly fixed slip angle, so a fixed ``vy`` range would miss it entirely at
    high speed.
    """
    p = p or VehicleParams()
    best = 0.0
    for alpha in np.linspace(0.0, 0.7, 350):
        # Rotate the velocity vector rather than adding to it, so total speed
        # -- and therefore downforce -- stays fixed across the sweep.
        vx, vy = speed * np.cos(alpha), speed * np.sin(alpha)
        state = np.array([[0.0, 0.0, 0.0, vx, vy, 0.0]])
        new, _af, _ar = step_dynamics(
            state,
            np.zeros(1),
            np.full(1, throttle),
            np.full(1, brake),
            p,
            1e-4,
        )
        # ay = dvy/dt + vx*r, and r stays 0 here.
        ay = (new[0, 4] - vy) / 1e-4
        best = max(best, abs(ay) * p.mass)
    return best


def test_aerodynamic_drag_is_not_charged_to_the_tyres():
    """Regression test for a bug that made the car spin at every corner entry.

    Drag acts on the body, not through the contact patch. Charging it to the
    friction ellipse costs the rear axle a fifth of its lateral grip at speed,
    which reads as inexplicable snap oversteer.
    """
    p = VehicleParams()
    speed = 80.0
    available = p.mu * (p.mass * G + p.k_down * speed**2)
    measured = peak_lateral_force(speed, throttle=0.0, p=p)
    # Not exactly equal: the two axles have different cornering stiffnesses so
    # they peak at slightly different slip angles and the sum loses a little.
    # Charging drag to the tyres would cost ~15% here, far outside this band.
    assert 0.94 * available < measured <= 1.01 * available


def test_longitudinal_demand_eats_lateral_grip():
    """Friction ellipse behaviour, in the directions the physics requires.

    Full throttle loads only the driven axle, so the front keeps its grip and
    the car still turns. Full braking spends the entire four-tyre budget on
    deceleration, and a tyre at its longitudinal limit has nothing left to
    corner with -- that is the ellipse, not a bug.
    """
    p = VehicleParams()
    coasting = peak_lateral_force(40.0, p=p)
    on_power = peak_lateral_force(40.0, throttle=1.0, p=p)
    braking = peak_lateral_force(40.0, brake=1.0, p=p)

    assert on_power < coasting, "power should cost rear grip"
    assert on_power > 0.4 * coasting, "the undriven front axle keeps its grip"
    assert braking < 0.4 * coasting, "braking at the limit should leave almost nothing"


def test_braking_force_cannot_exceed_available_grip():
    """The pedal must not be able to demand more than the tyres can transmit."""
    p = VehicleParams()
    for speed in (20.0, 45.0, 80.0):
        state = np.array([[0.0, 0.0, 0.0, speed, 0.0, 0.0]])
        new, _af, _ar = step_dynamics(
            state, np.zeros(1), np.zeros(1), np.ones(1), p, 1e-4
        )
        decel = (speed - new[0, 3]) / 1e-4
        grip_g = p.mu * (p.mass * G + p.k_down * speed**2) / p.mass
        brake_g = p.brake_force / p.mass
        limit = min(grip_g, brake_g) + p.k_drag * speed**2 / p.mass + 1.0
        assert decel <= limit, f"{decel:.1f} m/s^2 at {speed} m/s exceeds {limit:.1f}"


def test_downforce_raises_the_cornering_limit():
    p = VehicleParams()
    low = peak_lateral_force(20.0, p=p) / (p.mass * G)
    high = peak_lateral_force(80.0, p=p) / (p.mass * G)
    assert high > 2.5 * low


# ----------------------------------------------------------------------- env


@pytest.mark.parametrize("name", ["oval", "national", "gp", "technical"])
def test_reference_driver_completes_a_lap(name):
    env = make_env(name, n_envs=1, random_start=False, max_seconds=300.0)
    out = env.rollout(ReferenceDriver(half_width=env.track.half_width))
    assert out["retired"][0] == 0, f"retired on {name}"
    assert not np.isnan(out["lap_time"][0])
    avg_kph = env.track.length / out["lap_time"][0] * 3.6
    assert 100 < avg_kph < 300, f"implausible average speed {avg_kph:.0f} km/h"


def test_reference_driver_is_robust_to_random_starts():
    env = make_env("national", n_envs=64, random_start=True, max_seconds=120.0)
    out = env.rollout(ReferenceDriver(half_width=env.track.half_width), seed=3)
    assert np.mean(~np.isnan(out["lap_time"])) > 0.95


def test_retired_cars_stop_accruing_reward():
    env = make_env("technical", n_envs=8, random_start=False, max_seconds=30.0)
    env.reset(seed=0)
    hard_left = np.tile(np.array([1.0, 1.0, 0.0]), (8, 1))
    total = np.zeros(8)
    for _ in range(1500):
        _obs, reward, _done, _info = env.step(hard_left)
        total += reward
        if not env.alive.any():
            break
    assert not env.alive.any(), "driving into the wall should retire the car"
    frozen = env.progress.copy()
    for _ in range(50):
        _obs, reward, _done, _info = env.step(hard_left)
        assert np.allclose(reward, 0.0)
    assert np.allclose(env.progress, frozen)

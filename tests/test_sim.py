import numpy as np
import pytest

from flydrive.agents.classical import ReferenceDriver, corner_speed
from flydrive.sim import get_track, make_env
from flydrive.sim.track import (
    TRACKS,
    Track,
    from_layout,
    racing_line_offsets,
    ring_curvature,
    ring_normals,
)
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


# --------------------------------------------------------------- racing line


def _corridor(radius, half_width, n=900):
    """A closed circle plus the room either side of it."""
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    xy = np.stack([radius * np.cos(th), radius * np.sin(th)], axis=1)
    side = np.full(n, half_width)
    return xy, side


def test_racing_line_stays_inside_the_corridor():
    xy, side = _corridor(120.0, 5.0)
    alpha = racing_line_offsets(xy, side, side, margin=0.6)
    assert alpha.max() <= 5.0 - 0.6 + 1e-6
    assert alpha.min() >= -(5.0 - 0.6) - 1e-6


def test_racing_line_respects_asymmetric_room():
    """Bounds are per side, so a one-sided corridor must stay one-sided."""
    xy, _ = _corridor(120.0, 5.0)
    left = np.full(len(xy), 0.8)          # almost no room to the left
    right = np.full(len(xy), 6.0)
    alpha = racing_line_offsets(xy, left, right, margin=0.6)
    assert alpha.max() <= 0.2 + 1e-6
    assert alpha.min() >= -5.4 - 1e-6


def test_racing_line_opens_up_the_slowest_corner():
    """The point of the whole exercise: a bigger minimum radius.

    Minimising the plain integral of squared curvature does not achieve this --
    it is a lap-length average and will spend the hairpin to buy back a little
    on the sweepers. The solver raises the objective to a higher power for
    exactly this reason, so the test is on the minimum radius, not the mean.
    """
    circuit = get_track("national")
    side = np.full(circuit.n, 6.0)
    alpha = racing_line_offsets(circuit.xy, side, side, margin=0.6)
    line = circuit.xy + alpha[:, None] * ring_normals(circuit.xy)

    # Both measured the same way: a spline refit of one and the designed
    # curvature of the other would compare two different estimators.
    before = 1.0 / np.abs(ring_curvature(circuit.xy)).max()
    after = 1.0 / np.abs(ring_curvature(line)).max()
    assert after > 1.25 * before, f"minimum radius {before:.1f} m -> {after:.1f} m"


def test_racing_line_never_returns_something_worse_than_the_centreline():
    """The reweighting can cycle, so the routine must be able to decline.

    A circuit whose corners all turn the same way is the case where a
    minimum-curvature line has nothing to offer.
    """
    circuit = from_layout(
        [("straight", 300), ("corner", 30, -170), ("straight", 200),
         ("corner", 60, -100), ("straight", 260), ("corner", 45, -60),
         ("straight", 180), ("corner", 80, -30), ("straight", 220)],
        half_width=6.0,
    )
    side = np.full(circuit.n, 6.0)
    alpha = racing_line_offsets(circuit.xy, side, side, margin=0.6)
    line = circuit.xy + alpha[:, None] * ring_normals(circuit.xy)
    before = 1.0 / np.abs(ring_curvature(circuit.xy)).max()
    after = 1.0 / np.abs(ring_curvature(line)).max()
    assert after >= before - 1e-9, f"minimum radius {before:.1f} m -> {after:.1f} m"


def test_racing_line_track_is_asymmetric_but_covers_the_same_road():
    """Room lost on one side of the line has to reappear on the other."""
    circuit = get_track("gp")
    side = np.full(circuit.n, 6.0)
    alpha = racing_line_offsets(circuit.xy, side, side, margin=0.6)
    total = (side - alpha) + (side + alpha)
    assert np.allclose(total, 12.0)
    assert np.abs(alpha).max() > 2.0, "the line never left the centre"


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


def test_rear_axle_caps_the_brake_before_the_whole_car_does():
    """Corner entry is limited by the rear axle, not by the car's total grip.

    Comparing pedal travel against a grip *fraction* -- which is what the first
    version did -- let the driver ask for roughly twice the force the rear could
    take, and it spun into Spa's first corner every lap.
    """
    p = VehicleParams()
    drv = ReferenceDriver(params=p)
    vx = np.array([35.0])
    load = np.ones(1)
    straight = drv.rear_brake_limit(vx, np.zeros(1), load)
    cornering = drv.rear_brake_limit(vx, np.full(1, 14.0), load)

    # What the whole-car ellipse would have allowed at the same lateral load.
    grip = p.mu * (p.mass * G + p.k_down * vx**2)
    lat_frac = p.mass * 14.0 / grip[0]
    whole_car = grip[0] * np.sqrt(1.0 - lat_frac**2)

    assert cornering[0] < straight[0], "cornering must cost braking"
    assert cornering[0] < whole_car, "the rear axle is the binding limit"
    assert drv.rear_brake_limit(vx, np.full(1, 40.0), load)[0] == 0.0


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

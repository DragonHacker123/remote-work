"""The circuit test that the whole project rests on.

If PFL3R - PFL3L does not encode heading error as a sine with its steepest
slope at zero, there is no steering controller in here and the connectome is
just decoration.
"""

import numpy as np
import pytest

from flydrive.agents.net import THRESHOLD_BIAS, ConnectomeBrain, driving_subgraph
from flydrive.connectome import build_surrogate
from flydrive.sim.obs import OBS_DIM, RING_SLICE


def make_obs(heading, errors):
    obs = np.zeros((len(errors), OBS_DIM))
    goal = heading - errors
    obs[:, RING_SLICE] = np.column_stack(
        [
            np.full(len(errors), np.sin(heading)),
            np.full(len(errors), np.cos(heading)),
            np.sin(goal),
            np.cos(goal),
        ]
    )
    return obs


def settle(brain, obs, steps=80):
    brain.reset(obs.shape[0])
    action = None
    for _ in range(steps):
        action = brain(obs)
    return action


@pytest.fixture(scope="module")
def brain():
    return ConnectomeBrain(driving_subgraph(build_surrogate(scale=0.5, seed=0)))


def test_epg_bump_tracks_heading(brain):
    """The compass bump sits at the heading the animal is pointing."""
    for heading in (0.0, 1.2, -2.5):
        obs = make_obs(heading, np.array([0.0]))
        settle(brain, obs, steps=60)
        rates = brain.rates("heading")[:, 0]
        wedges = np.linspace(0, 2 * np.pi, len(rates), endpoint=False)
        peak = wedges[int(np.argmax(rates))]
        offset = np.angle(np.exp(1j * (peak - heading)))
        assert abs(offset) < 2 * np.pi / len(rates), f"bump {offset:.3f} rad off heading"


def test_pfl3_difference_is_sinusoidal_in_heading_error(brain):
    errors = np.linspace(-np.pi, np.pi, 49)
    settle(brain, make_obs(0.7, errors))
    diff = brain.rates("PFL3R").mean(axis=0) - brain.rates("PFL3L").mean(axis=0)
    r = np.corrcoef(diff, np.sin(errors))[0, 1]
    assert abs(r) > 0.9, f"PFL3 difference is not a sine of heading error (r={r:.3f})"


def test_pfl3_signal_is_steepest_at_zero_error(brain):
    """Maximum sensitivity where it matters -- small corrections.

    Restricted to |error| < pi/2: a sine's slope also peaks at +/-pi, so
    sweeping the full circle would let the test pass on the wrong extremum.
    """
    errors = np.linspace(-np.pi / 2, np.pi / 2, 41)
    settle(brain, make_obs(0.7, errors))
    diff = brain.rates("PFL3R").mean(axis=0) - brain.rates("PFL3L").mean(axis=0)
    grad = np.abs(np.gradient(diff, errors))
    assert np.argmax(grad) in range(len(errors) // 2 - 4, len(errors) // 2 + 5)


def test_steering_output_opposes_heading_error(brain):
    """Sign check: pointing left of the goal must command a right turn."""
    errors = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    action = settle(brain, make_obs(0.7, errors))
    steer = action[:, 0]
    assert np.corrcoef(steer, errors)[0, 1] < -0.9, (
        "steering does not correct the heading error"
    )


def test_readout_is_invariant_to_absolute_heading(brain):
    """A ring attractor must report error, not compass direction.

    If this fails the ring has lost translation invariance, and absolute
    heading -- which sweeps a full turn every lap -- will swamp the few degrees
    of steering error the circuit is supposed to report.
    """
    errors = np.linspace(-0.6, 0.6, 13)
    curves = []
    for heading in (0.0, 1.6, 3.0, -2.2):
        settle(brain, make_obs(heading, errors))
        curves.append(
            brain.rates("PFL3R").mean(axis=0) - brain.rates("PFL3L").mean(axis=0)
        )
    curves = np.array(curves)
    spread = curves.std(axis=0).mean()
    signal = np.ptp(curves.mean(axis=0))
    assert spread < 0.4 * signal, f"heading-dependent drift {spread:.4f} vs signal {signal:.4f}"


def test_threshold_bias_is_what_creates_the_signal():
    """Above threshold the rectified population sum degenerates to a constant.

    This is the single most important initialisation constant in the model, so
    it gets a test that fails loudly if someone 'tidies it up'.
    """
    conn = driving_subgraph(build_surrogate(scale=0.5, seed=0))
    brain = ConnectomeBrain(conn)
    errors = np.linspace(-np.pi, np.pi, 49)
    obs = make_obs(0.7, errors)
    bias_slice = brain.layout.slices["bias"]

    def slope_at_zero(bias_value):
        theta = brain.initial_params().copy()
        bias = theta[bias_slice].copy()
        for name in ("PFL3L", "PFL3R"):
            bias[conn.type_index(name)] = bias_value
        theta[bias_slice] = bias
        brain.set_params(theta)
        settle(brain, obs)
        diff = brain.rates("PFL3R").mean(axis=0) - brain.rates("PFL3L").mean(axis=0)
        mid = len(errors) // 2
        return abs((diff[mid + 1] - diff[mid - 1]) / (errors[mid + 1] - errors[mid - 1]))

    assert slope_at_zero(THRESHOLD_BIAS) > 20 * slope_at_zero(+0.05)

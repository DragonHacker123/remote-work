"""End-to-end checks on the two claims the project actually makes."""

import numpy as np
import pytest

from flydrive.agents.classical import PurePursuitDriver
from flydrive.agents.net import ConnectomeBrain, driving_subgraph
from flydrive.connectome import build_surrogate
from flydrive.learn.distill import fit_readout, imitation_loss
from flydrive.learn.mb import MBConfig, MushroomBody, run_learning_session
from flydrive.sim import make_env


def test_mushroom_body_learns_a_context_dependent_output():
    """The three-factor rule, tested precisely and without a car in the way.

    Each sector has its own correct motor trim. Dopamine reports only whether a
    perturbation helped, so the circuit has to discover the mapping by
    exploration -- and the sparse Kenyon-cell code has to keep each sector's
    memory from overwriting its neighbours'.
    """
    conn = build_surrogate(scale=0.5, seed=0)
    cfg = MBConfig(n_sectors=16, sparsity=8, lr=0.4, seed=0)
    mb = MushroomBody(conn, cfg)
    rng = np.random.default_rng(0)
    # Targets must sit inside the trim's deliberate bound (weight_clip x
    # trim_scale): the MBON output is a correction to a working driver, not a
    # controller, and asking for trims it cannot reach tests nothing.
    reach = cfg.weight_clip * cfg.trim_scale
    target = rng.uniform(-0.8 * reach, 0.8 * reach, (cfg.n_sectors, MushroomBody.N_OUTPUTS))

    sectors_all = np.arange(cfg.n_sectors, dtype=float)
    start_error = np.abs(mb.output(mb.kc_code(sectors_all)) - target).mean()

    for _ in range(4000):
        sectors = rng.integers(0, cfg.n_sectors, size=8)
        code = mb.kc_code(sectors.astype(float))
        baseline = mb.output(code)
        perturb = mb.perturb(len(sectors))
        want = target[sectors]
        # Dopamine: did the perturbation move us closer to the right trim?
        reward = np.abs(baseline - want).sum(1) - np.abs(baseline + perturb - want).sum(1)
        mb.update(code, perturb, reward / cfg.explore)

    end_error = np.abs(mb.output(mb.kc_code(sectors_all)) - target).mean()
    assert end_error < 0.55 * start_error, f"{start_error:.3f} -> {end_error:.3f}"


@pytest.mark.slow
def test_mushroom_body_improves_lap_times_with_inherited_weights_frozen():
    """Ontogeny: the same driver gets faster lap after lap.

    Nothing in the inherited network changes here -- the entire improvement
    lives in KC->MBON synapses updated by a dopamine-gated three-factor rule,
    with each sector's time judged against the best the car has managed there.

    Run with many cars sharing one mushroom body. That is a variance-reduction
    trick rather than biology (a single fly has one brain and would need far
    more laps); the learning rule is the same either way.
    """
    conn = build_surrogate(scale=1.0, seed=0)
    laps = 16
    env = make_env(
        "national", n_envs=24, random_start=False, max_seconds=laps * 80.0, target_laps=laps + 1
    )
    base = PurePursuitDriver(half_width=env.track.half_width)

    results = {}
    for learn in (False, True):
        mb = MushroomBody(conn, MBConfig(n_sectors=24, sparsity=10, seed=1))
        out = run_learning_session(env, base, mb, laps=laps, learn=learn, seed=2)
        assert out["laps_completed"] == laps, "session did not finish its laps"
        assert out["alive"].all(), "plasticity should not put the car in the wall"
        results[learn] = np.nanmean(out["lap_times"], axis=1)

    frozen, learned = results[False], results[True]
    # Without plasticity the driver is deterministic, so lap time is flat.
    assert np.ptp(frozen[1:]) < 0.05, f"frozen baseline drifted: {frozen}"
    # With it, the last laps beat both the first laps and the frozen baseline.
    assert learned[-3:].mean() < learned[:3].mean() - 0.5, f"no improvement: {learned}"
    assert learned[-3:].mean() < frozen[1:].mean(), "learned laps not faster than baseline"


@pytest.mark.slow
def test_distillation_makes_the_network_drive_further_than_it_starts():
    """Phylogeny, stage one: the readout fit has to actually buy something.

    Run at full scale, because the gain from distillation shrinks with the
    network -- a small connectome has little for the readout to exploit.
    """
    conn = driving_subgraph(build_surrogate(scale=1.0, seed=0))
    brain = ConnectomeBrain(conn)
    train_env = make_env("national", n_envs=12, max_seconds=30.0)
    eval_env = make_env("national", n_envs=12, max_seconds=60.0)

    before = eval_env.rollout(brain, seed=1)["progress"].mean()
    fit = fit_readout(brain, train_env, steps=1500, seed=0)
    after = eval_env.rollout(brain, seed=1)["progress"].mean()

    assert fit["r2_steer"] > 0.6, f"steering barely fits: {fit}"
    assert after > 1.8 * before, f"{before:.0f} m -> {after:.0f} m"


def test_imitation_loss_prefers_the_teacher_to_a_scrambled_copy():
    """Sanity check on the training objective itself."""
    conn = driving_subgraph(build_surrogate(scale=0.5, seed=0))
    brain = ConnectomeBrain(conn)
    env = make_env("national", n_envs=8, max_seconds=20.0)
    fit_readout(brain, env, steps=800, seed=0)

    good = imitation_loss(brain, env, steps=400, seed=0)
    scrambled = ConnectomeBrain(conn)
    rng = np.random.default_rng(0)
    scrambled.set_params(brain.theta + rng.normal(0.0, 1.5, brain.n_params))
    bad = imitation_loss(scrambled, env, steps=400, seed=0)
    assert good < bad, f"distilled {good:.4f} should beat scrambled {bad:.4f}"


@pytest.mark.parametrize("control", ["none", "within-type"])
def test_pipeline_runs_on_real_and_shuffled_connectomes(control):
    """Both arms of the control experiment must be runnable end to end."""
    from flydrive.connectome import apply_control

    conn = apply_control(build_surrogate(scale=0.35, seed=0), control, seed=1)
    brain = ConnectomeBrain(driving_subgraph(conn))
    env = make_env("national", n_envs=8, max_seconds=15.0)
    fit = fit_readout(brain, env, steps=600, seed=0)
    assert np.isfinite(fit["r2_steer"])
    out = env.rollout(brain, seed=1)
    assert np.all(np.isfinite(out["progress"]))

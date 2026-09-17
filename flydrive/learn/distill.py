"""Bootstrap the brain by distilling the classical driver into its readout.

Evolution strategies cannot get started from a policy that crashes in the first
second: every candidate scores the same, so there is no gradient to climb. The
fix is to hand ES a starting point that already drives.

The recurrent dynamics are fixed by the connectome, so the network is run under
*teacher forcing* -- the car is driven by the reference controller while the
network watches the same observations -- and only the linear decoder from
descending-neuron rates to controls is fitted, in closed form by ridge
regression. This is the standard reservoir-computing readout fit, and it is
also the biologically plausible half: the recurrent circuit is inherited, the
premotor synapses are plastic.

The R^2 of that fit is worth reading on its own. It answers the question that
decides whether this whole approach can work: **how much of a competent driving
policy is linearly available in the fly's descending-neuron population?**
"""

from __future__ import annotations

import numpy as np

from ..agents.classical import ReferenceDriver
from ..agents.net import ConnectomeBrain
from ..sim.env import RaceEnv


def collect(
    brain: ConnectomeBrain,
    env: RaceEnv,
    teacher=None,
    steps: int = 1500,
    seed: int = 0,
    student_frac: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the teacher, recording (DN rates, teacher action) pairs.

    Only samples where the car is still running are kept -- a retired car's
    frozen state is not a driving demonstration.

    ``student_frac`` blends the student into what is driven. Zero for the first
    fit, when the student is untrained and would just wreck the car; non-zero
    when re-fitting a partly trained network, so the readout is fitted on the
    states it will actually encounter rather than only the teacher's line.
    """
    teacher = teacher or ReferenceDriver(params=env.cfg.params, half_width=env.track.half_width)
    obs = env.reset(seed=seed)
    brain.reset(env.n)
    if hasattr(teacher, "reset"):
        teacher.reset(env.n)

    feats: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for _ in range(steps):
        action = teacher(obs)
        pred = brain(obs)                # network watches, teacher drives
        live = env.alive.copy()
        if live.any():
            feats.append(brain.readout_rates()[live])
            targets.append(action[live])
        driven = action if student_frac <= 0 else (
            (1.0 - student_frac) * action + student_frac * pred
        )
        obs, _r, _d, _info = env.step(driven)
        if not env.alive.any():
            break
    if not feats:
        raise RuntimeError("teacher retired immediately; nothing to distil from")
    return np.concatenate(feats, axis=0), np.concatenate(targets, axis=0)


def imitation_loss(
    brain: ConnectomeBrain,
    env: RaceEnv,
    teacher=None,
    steps: int = 900,
    seed: int = 0,
    weights: tuple[float, float, float] = (3.0, 1.0, 1.0),
    student_frac: float = 0.5,
) -> float:
    """Mean squared error against the teacher under teacher forcing.

    A dense objective, unlike driving reward: every timestep scores, whether or
    not the network could have kept the car on the road. That is what lets ES
    make progress from a starting point that cannot complete a corner, and it
    is the only objective that can fix the *encoder* -- a closed-form readout
    fit takes the sensory projection as given, however bad it is.

    Steering is weighted up because it is the channel that decides whether the
    car survives the corner at all.

    ``student_frac`` blends the student's action into what is actually driven,
    DAgger-style. Under pure teacher forcing the network only ever sees states
    on the teacher's line, so it is never scored on recovering from its own
    mistakes -- and its first mistake in closed loop takes it somewhere it has
    no training signal for. Letting it drift and still labelling with the
    teacher is what closes that gap.
    """
    teacher = teacher or ReferenceDriver(
        params=env.cfg.params, half_width=env.track.half_width
    )
    obs = env.reset(seed=seed)
    brain.reset(env.n)
    if hasattr(teacher, "reset"):
        teacher.reset(env.n)

    w = np.asarray(weights)
    total, count = 0.0, 0
    for _ in range(steps):
        target = teacher(obs)
        pred = brain(obs)
        live = env.alive
        if live.any():
            diff = pred[live] - target[live]
            total += float((w * diff * diff).sum())
            count += int(diff.shape[0])
        driven = (1.0 - student_frac) * target + student_frac * pred
        obs, _r, _d, _info = env.step(driven)
        if not env.alive.any():
            break
    return total / max(count, 1)


def fit_readout(
    brain: ConnectomeBrain,
    env: RaceEnv,
    teacher=None,
    steps: int = 1500,
    ridge: float = 1e-3,
    seed: int = 0,
    student_frac: float = 0.0,
) -> dict:
    """Least-squares fit the decoder to the teacher. Returns fit diagnostics."""
    x, y = collect(brain, env, teacher, steps, seed, student_frac)

    # Invert the output squashing so the regression targets are pre-activation.
    # Throttle and brake collapse into one signed longitudinal target, matching
    # the decoder's two channels.
    z = np.column_stack(
        [
            np.arctanh(np.clip(y[:, 0], -0.999, 0.999)),
            y[:, 1] - y[:, 2],
        ]
    )

    design = np.column_stack([x, np.ones(len(x))])
    gram = design.T @ design
    gram[np.diag_indices_from(gram)] += ridge * len(x)
    coef = np.linalg.solve(gram, design.T @ z)

    pred = design @ coef
    resid = ((z - pred) ** 2).sum(axis=0)
    total = ((z - z.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - resid / np.maximum(total, 1e-12)

    brain.set_readout(coef[:-1], coef[-1])
    return {
        "r2_steer": float(r2[0]),
        "r2_longitudinal": float(r2[1]),
        "samples": int(len(x)),
        "readout_dim": int(x.shape[1]),
    }

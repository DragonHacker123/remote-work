"""Outer training loop: fit the free parameters of a connectome brain by ES.

Fitness is the return over a *fixed horizon* from randomised starting points.
With the horizon fixed, maximising progress is the same thing as maximising
average speed, so this optimises lap time directly without needing the car to
complete a lap before it gets any signal.

All candidates in a generation are evaluated from the *same* starting states
(common random numbers). Without that, the difference between two candidates
is swamped by the difference between an easy start and a hard one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np

from ..agents.classical import ReferenceDriver
from ..agents.net import BrainConfig, ConnectomeBrain, driving_subgraph
from ..connectome import apply_control, build_surrogate
from ..sim import make_env
from .distill import fit_readout, imitation_loss
from .es import ESConfig, OpenAIES


@dataclass
class TrainConfig:
    track: str = "national"
    n_envs: int = 32
    horizon_seconds: float = 25.0
    generations: int = 60
    workers: int = 0                   # 0 or 1 = serial
    scale: float = 1.0
    control: str = "none"              # connectome null model, see connectome.controls
    connectome_seed: int = 0
    seed: int = 0
    log_every: int = 1
    objective: str = "reward"          # "reward" or "imitation"
    es: ESConfig = field(default_factory=ESConfig)


def build_brain(cfg: TrainConfig) -> ConnectomeBrain:
    """Construct the driving network, optionally over a shuffled connectome."""
    conn = build_surrogate(scale=cfg.scale, seed=cfg.connectome_seed)
    conn = apply_control(conn, cfg.control, seed=cfg.connectome_seed + 1)
    return ConnectomeBrain(driving_subgraph(conn), BrainConfig(seed=cfg.seed))


# --------------------------------------------------------------------- worker

_W: dict = {}


def _init_worker(cfg: TrainConfig) -> None:
    _W["cfg"] = cfg
    _W["brain"] = build_brain(cfg)
    env = make_env(
        cfg.track, n_envs=cfg.n_envs, seed=cfg.seed, max_seconds=cfg.horizon_seconds
    )
    _W["env"] = env
    _W["teacher"] = ReferenceDriver(
        params=env.cfg.params, half_width=env.track.half_width
    )
    _W["horizon"] = int(cfg.horizon_seconds * env.cfg.control_hz)


def _evaluate(args) -> tuple[float, float, float]:
    theta, gen_seed = args
    cfg, brain, env = _W["cfg"], _W["brain"], _W["env"]
    brain.set_params(theta)

    if cfg.objective == "imitation":
        loss = imitation_loss(
            brain, env, _W["teacher"], steps=_W["horizon"], seed=gen_seed
        )
        return -loss, 0.0, 0.0

    out = env.rollout(brain, horizon=_W["horizon"], seed=gen_seed)
    laps = float(np.mean(~np.isnan(out["lap_time"])))
    return float(out["return"].mean()), float(out["progress"].mean()), laps


# ---------------------------------------------------------------------- train


def train(cfg: TrainConfig, callback=None, theta0: np.ndarray | None = None) -> dict:
    """Run ES and return the best parameters found plus a training log."""
    probe = build_brain(cfg)
    start = probe.initial_params() if theta0 is None else np.asarray(theta0)
    optimizer = OpenAIES(start, cfg.es)
    rng = np.random.default_rng(cfg.seed + 7919)

    pool = None
    if cfg.workers and cfg.workers > 1:
        import multiprocessing as mp

        pool = mp.get_context("fork").Pool(
            cfg.workers, initializer=_init_worker, initargs=(cfg,)
        )
    else:
        _init_worker(cfg)

    log: list[dict] = []
    best = {"fitness": -np.inf, "theta": optimizer.theta.copy()}
    started = time.time()
    try:
        for gen in range(cfg.generations):
            candidates = optimizer.ask()
            gen_seed = int(rng.integers(1 << 30))
            work = [(c, gen_seed) for c in candidates]
            results = pool.map(_evaluate, work) if pool else [_evaluate(w) for w in work]
            fitness = np.array([r[0] for r in results])
            progress = np.array([r[1] for r in results])
            laps = np.array([r[2] for r in results])
            optimizer.tell(fitness)

            top = int(np.argmax(fitness))
            if fitness[top] > best["fitness"]:
                best = {"fitness": float(fitness[top]), "theta": candidates[top].copy()}

            row = {
                "gen": gen,
                "fit_mean": float(fitness.mean()),
                "fit_max": float(fitness.max()),
                "progress_mean": float(progress.mean()),
                "progress_max": float(progress.max()),
                "lap_frac": float(laps.max()),
                "sigma": optimizer.sigma,
                "elapsed": time.time() - started,
            }
            log.append(row)
            if callback:
                callback(row)
            elif cfg.log_every and gen % cfg.log_every == 0:
                print(
                    f"gen {gen:3d}  fit {row['fit_mean']:8.1f} (max {row['fit_max']:8.1f})"
                    f"  progress {row['progress_mean']:6.0f} m (max {row['progress_max']:6.0f})"
                    f"  laps {row['lap_frac']:.2f}  sigma {row['sigma']:.3f}"
                    f"  {row['elapsed']:6.0f}s",
                    flush=True,
                )
    finally:
        if pool:
            pool.close()
            pool.join()

    return {
        "theta": optimizer.theta,
        "best_theta": best["theta"],
        "best_fitness": best["fitness"],
        "log": log,
        "config": cfg,
    }


def train_curriculum(
    cfg: TrainConfig,
    imitation_generations: int = 40,
    reward_generations: int = 80,
    callback=None,
) -> dict:
    """The full three-stage pipeline.

    1. Fit the descending-neuron readout to the reference driver in closed
       form. Instant, and it establishes how much of a driving policy is
       linearly available in the motor bus at all.
    2. ES on imitation loss. Dense signal, and unlike step 1 it can reshape the
       sensory encoder -- which is what actually limits steering accuracy.
    3. ES on driving reward over a fixed horizon, which is lap-time
       optimisation. Only this stage can beat the teacher, because only here is
       the objective speed rather than similarity.
    """
    brain = build_brain(cfg)
    env = make_env(
        cfg.track, n_envs=cfg.n_envs, seed=cfg.seed, max_seconds=cfg.horizon_seconds
    )
    stages: list[dict] = []

    fit = fit_readout(brain, env, steps=int(cfg.horizon_seconds * 50), seed=cfg.seed)
    theta = brain.theta.copy()
    stages.append({"stage": "readout_fit", **fit})
    if cfg.log_every:
        print(f"[1/3] readout fit: {fit}", flush=True)

    if imitation_generations > 0:
        icfg = replace(cfg, objective="imitation", generations=imitation_generations)
        res = train(icfg, callback=callback, theta0=theta)
        theta = res["theta"]
        stages.append({"stage": "imitation", "log": res["log"]})
        # Re-fit the readout on the reshaped encoder: it is free and exact.
        brain.set_params(theta)
        fit2 = fit_readout(
            brain, env, steps=int(cfg.horizon_seconds * 50), seed=cfg.seed,
            student_frac=0.5,   # refit on states the student actually reaches
        )
        theta = brain.theta.copy()
        stages.append({"stage": "readout_refit", **fit2})
        if cfg.log_every:
            print(f"[2/3] readout refit: {fit2}", flush=True)

    res = train(
        replace(cfg, objective="reward", generations=reward_generations),
        callback=callback,
        theta0=theta,
    )
    stages.append({"stage": "reward", "log": res["log"]})
    return {
        "theta": res["theta"],
        "best_theta": res["best_theta"],
        "stages": stages,
        "config": cfg,
    }


def evaluate_laps(
    theta: np.ndarray, cfg: TrainConfig, track: str | None = None, n_envs: int = 32
) -> dict:
    """Measure true lap times from the start line with a fixed parameter set."""
    brain = build_brain(cfg)
    brain.set_params(theta)
    env = make_env(
        track or cfg.track, n_envs=n_envs, seed=cfg.seed + 1, max_seconds=240.0,
        random_start=False,
    )
    out = env.rollout(brain, seed=cfg.seed + 1)
    laps = out["lap_time"]
    done = ~np.isnan(laps)

    # Where and how the car gives up is more useful than the bare distance: a
    # policy that always dies at the same metre mark has one specific problem,
    # not a general lack of skill.
    reasons = {0: "finished", 1: "off-track", 2: "spun", 3: "stalled"}
    counts = np.bincount(out["retired"], minlength=4)
    return {
        "completed": int(done.sum()),
        "n": n_envs,
        "lap_time": float(np.nanmin(laps)) if done.any() else float("nan"),
        "mean_lap": float(np.nanmean(laps)) if done.any() else float("nan"),
        "progress": float(out["progress"].mean()),
        "progress_max": float(out["progress"].max()),
        "track_length": env.track.length,
        "retired": {reasons[i]: int(c) for i, c in enumerate(counts) if c},
    }

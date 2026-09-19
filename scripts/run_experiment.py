"""Train a connectome brain to drive, then repeat on a shuffled control.

The control is the point. A network built on real fly wiring that cannot beat
a null model matching its degree sequence and cell-type composition has not
demonstrated anything about connectomes.

    python scripts/run_experiment.py --generations 250 --out results/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from flydrive.agents.classical import ReferenceDriver
from flydrive.learn.es import ESConfig
from flydrive.learn.train import TrainConfig, evaluate_laps, train_curriculum
from flydrive.sim import make_env


def reference_lap(track: str) -> dict:
    env = make_env(track, n_envs=8, random_start=False, max_seconds=300.0)
    out = env.rollout(ReferenceDriver(half_width=env.track.half_width))
    laps = out["lap_time"]
    return {
        "lap_time": float(np.nanmin(laps)) if np.any(~np.isnan(laps)) else float("nan"),
        "track_length": float(env.track.length),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="spa")
    ap.add_argument("--generations", type=int, default=250)
    ap.add_argument("--imitation", type=int, default=30)
    ap.add_argument("--popsize", type=int, default=24)
    ap.add_argument("--envs", type=int, default=24)
    ap.add_argument("--horizon", type=float, default=18.0)
    ap.add_argument("--workers", type=int, default=4)
    # One seed per arm is a single sample of a stochastic pipeline. Large
    # effects here can still be seed variance, so vary this and pool.
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--controls", default="none,within-type")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = reference_lap(args.track)
    print(f"reference driver on {args.track}: {ref['lap_time']:.2f}s "
          f"over {ref['track_length']:.0f} m", flush=True)

    summary = {"track": args.track, "reference": ref, "conditions": {}}
    for control in args.controls.split(","):
        control = control.strip()
        print(f"\n{'='*70}\ncondition: connectome control = {control!r}\n{'='*70}", flush=True)
        cfg = TrainConfig(
            track=args.track,
            n_envs=args.envs,
            horizon_seconds=args.horizon,
            workers=args.workers,
            control=control,
            seed=args.seed,
            log_every=10,
            es=ESConfig(popsize=args.popsize, sigma=0.08, lr=0.05, seed=args.seed),
        )
        started = time.time()
        res = train_curriculum(
            cfg, imitation_generations=args.imitation, reward_generations=args.generations
        )
        # Evaluate both the population mean and the single best candidate. The
        # mean is usually the better policy -- a candidate can top one
        # generation on a lucky set of starting states -- but not always, so
        # report both rather than picking one blind.
        laps_mean = evaluate_laps(res["theta"], cfg)
        laps_best = evaluate_laps(res["best_theta"], cfg)
        laps = max(
            (laps_mean, laps_best),
            key=lambda r: (r["completed"], r["progress"]),
        )
        elapsed = time.time() - started
        print(f"\n[{control}] mean={laps_mean}", flush=True)
        print(f"[{control}] best={laps_best}  ({elapsed:.0f}s)", flush=True)

        np.savez(
            out_dir / f"theta_{control}.npz",
            theta=res["theta"],
            best_theta=res["best_theta"],
        )
        reward_log = next(s["log"] for s in res["stages"] if s["stage"] == "reward")
        summary["conditions"][control] = {
            "laps": laps,
            "laps_mean_theta": laps_mean,
            "laps_best_theta": laps_best,
            "elapsed": elapsed,
            "readout_fit": next(
                s for s in res["stages"] if s["stage"] == "readout_fit"
            ),
            "readout_refit": next(
                (s for s in res["stages"] if s["stage"] == "readout_refit"), None
            ),
            "final_progress": reward_log[-1]["progress_mean"],
            "best_progress": max(r["progress_max"] for r in reward_log),
            "curve": [
                {k: r[k] for k in ("gen", "fit_mean", "progress_mean")}
                for r in reward_log[::10]
            ],
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nwrote {out_dir/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()

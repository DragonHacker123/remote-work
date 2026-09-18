"""Train the connectome brain, snapshotting it along the way, then replay every
snapshot so the learning itself can be watched.

Output is one JSON trace: the ES fitness curve, and for each checkpoint a lap
attempt with the car's pose and the firing rates of the circuit that steers it.
Early checkpoints crash at the first corner; late ones complete the lap.

    python scripts/record_training.py --out training.json
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np

from flydrive.agents.classical import ReferenceDriver
from flydrive.agents.net import ConnectomeBrain, driving_subgraph
from flydrive.connectome import apply_control, build_surrogate
from flydrive.learn.es import ESConfig
from flydrive.learn.train import TrainConfig, train_curriculum
from flydrive.sim import make_env
from flydrive.sim.obs import V_SCALE

# Populations drawn per neuron by the visualiser. Keyed by output name because
# the ports are called "heading" and "goal", and those words already mean the
# car's own pose here.
PER_NEURON = {
    "epg": "heading",     # compass ring
    "fc2": "goal",        # goal ring
    "pfl3l": "PFL3L",
    "pfl3r": "PFL3R",
    "d7": "Delta7",
}
MEANS = {
    "lal_l": "LAL_L", "lal_r": "LAL_R",
    "dna02_l": "DNa02_L", "dna02_r": "DNa02_R",
    "dna01": "speed", "dnp09": "stop",
    "hs": "HS_L", "lplc2": "LPLC2", "an": "AN_L",
}


def replay(theta, conn, track_name, max_frames, stride, seed):
    """Run one attempt and record pose plus neural state."""
    brain = ConnectomeBrain(conn)
    brain.set_params(theta)
    env = make_env(track_name, n_envs=1, random_start=False, max_seconds=300.0)
    obs = env.reset(seed=seed)
    brain.reset(1)

    out = {k: [] for k in ("x", "y", "heading", "speed", "steer", "throttle", "brake", "e_y")}
    for key in list(PER_NEURON) + list(MEANS):
        out[key] = []

    step = 0
    limit = int(300.0 / env.dt)
    while step < limit and len(out["x"]) < max_frames:
        action = brain(obs)
        if step % stride == 0:
            out["x"].append(float(env.state[0, 0]))
            out["y"].append(float(env.state[0, 1]))
            out["heading"].append(float(env.state[0, 2]))
            out["speed"].append(float(obs[0, 0] * V_SCALE))
            out["steer"].append(float(action[0, 0]))
            out["throttle"].append(float(action[0, 1]))
            out["brake"].append(float(action[0, 2]))
            out["e_y"].append(float(env.e_y[0]))
            for key, port in PER_NEURON.items():
                out[key].append(brain.rates(port)[:, 0].copy())
            for key, port in MEANS.items():
                out[key].append(float(brain.rates(port)[:, 0].mean()))
        obs, _r, _done, _info = env.step(action)
        step += 1
        if not env.alive.any():
            break

    lap = float(env.lap_time[0])
    return {
        "frames": out,
        "progress": float(env.progress[0]),
        "retired": int(env.retired_reason[0]),
        "lap_time": None if np.isnan(lap) else round(lap, 2),
        "seconds": round(step * env.dt, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="spa")
    ap.add_argument("--generations", type=int, default=500)
    ap.add_argument("--imitation", type=int, default=50)
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--lap-eval-every", type=int, default=25)
    ap.add_argument("--popsize", type=int, default=24)
    ap.add_argument("--envs", type=int, default=20)
    ap.add_argument("--horizon", type=float, default=18.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--control", default="none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=2400)
    ap.add_argument("--out", default="training.json")
    args = ap.parse_args()

    cfg = TrainConfig(
        track=args.track,
        n_envs=args.envs,
        horizon_seconds=args.horizon,
        workers=args.workers,
        control=args.control,
        seed=args.seed,
        log_every=10,
        checkpoint_every=args.every,
        lap_eval_every=args.lap_eval_every,
        es=ESConfig(popsize=args.popsize, sigma=0.08, lr=0.05, seed=args.seed),
    )
    res = train_curriculum(
        cfg, imitation_generations=args.imitation, reward_generations=args.generations
    )
    print(f"\ntraining done, {len(res['checkpoints'])} checkpoints -- replaying", flush=True)

    conn = driving_subgraph(
        apply_control(build_surrogate(seed=0), args.control, seed=1)
    )
    replays = []
    for i, ck in enumerate(res["checkpoints"]):
        r = replay(ck["theta"], conn, args.track, args.max_frames, args.stride, args.seed + 1)
        r.update(stage=ck["stage"], gen=ck["gen"], fit_mean=ck["fit_mean"])
        replays.append(r)
        print(f"  [{i+1}/{len(res['checkpoints'])}] {ck['stage']:10s} gen {ck['gen']:3d} "
              f"-> {r['progress']:6.0f} m, lap {r['lap_time']}", flush=True)

    # Normalise each population against its maximum across the WHOLE run, not
    # per checkpoint: the point is to see activity organise over training, and
    # per-checkpoint scaling would hide exactly that.
    scales = {}
    for key in list(PER_NEURON) + list(MEANS):
        vals = [np.asarray(r["frames"][key]) for r in replays if r["frames"][key]]
        scales[key] = float(max((v.max() for v in vals if v.size), default=1.0)) or 1.0

    def quant(arr, scale):
        """Quantise to bytes and base64 it.

        Spa laps are three times longer than the old circuit, so the raw JSON
        numbers would be most of a ten-megabyte page. One byte per neuron per
        frame, base64-encoded, is about a third of the size and plenty of
        resolution for a colour ramp.
        """
        b = np.clip(np.round(np.asarray(arr) / scale * 100.0), 0, 100).astype(np.uint8)
        return base64.b64encode(b.tobytes()).decode("ascii")

    for r in replays:
        f = r["frames"]
        for key in ("x", "y"):
            f[key] = np.round(f[key], 1).tolist()
        f["heading"] = np.round(f["heading"], 3).tolist()
        f["speed"] = np.round(f["speed"], 1).tolist()
        f["e_y"] = np.round(f["e_y"], 2).tolist()
        for key in ("steer", "throttle", "brake"):
            f[key] = np.round(f[key], 3).tolist()
        for key in PER_NEURON:
            f[key] = quant(np.asarray(f[key]).ravel(), scales[key])
        for key in MEANS:
            f[key] = quant(f[key], scales[key])

    # Reference driver's racing line, for comparison.
    ref_env = make_env(args.track, n_envs=1, random_start=False, max_seconds=300.0)
    ref_obs = ref_env.reset(seed=args.seed + 1)
    driver = ReferenceDriver(half_width=ref_env.track.half_width)
    line = []
    for _ in range(int(300.0 / ref_env.dt)):
        line.append([float(ref_env.state[0, 0]), float(ref_env.state[0, 1])])
        ref_obs, _r, _d, _i = ref_env.step(driver(ref_obs))
        if not ref_env.alive.any():
            break

    track = ref_env.track
    curves = {
        s["stage"]: [
            {"gen": row["gen"], "fit": round(row["fit_mean"], 2),
             "progress": round(row["progress_mean"], 1)}
            for row in s["log"]
        ]
        for s in res["stages"] if "log" in s
    }

    payload = {
        "track": {
            "xy": np.round(track.xy[::3], 1).tolist(),
            "half_width": track.half_width,
            "length": round(track.length, 1),
            "name": args.track,
        },
        "reference": {
            "line": np.round(np.array(line)[::8], 1).tolist(),
            "lap_time": round(float(ref_env.lap_time[0]), 2),
        },
        "dt": round(ref_env.dt * args.stride, 4),
        "control": args.control,
        "curves": curves,
        "checkpoints": replays,
        "populations": {
            "per_neuron": list(PER_NEURON),
            "means": list(MEANS),
        },
        "widths": {"epg": 16, "fc2": 16, "pfl3l": 16, "pfl3r": 16, "d7": 8},
        "encoding": "base64-uint8, 0-100, row-major (frame, neuron)",
        "fastest_lap": res.get("fastest_lap"),
        "lap_curve": [
            {"gen": r["gen"], "lap": r["lap_time"]}
            for st in res["stages"] if "log" in st
            for r in st["log"] if r.get("lap_time")
        ],
    }
    Path(args.out).write_text(json.dumps(payload, separators=(",", ":")))
    print(f"\nwrote {args.out} ({Path(args.out).stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()

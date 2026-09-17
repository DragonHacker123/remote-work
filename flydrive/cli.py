"""Command line entry point: ``flydrive <command>``."""

from __future__ import annotations

import argparse
import sys

import numpy as np


def _load_theta(path: str) -> np.ndarray:
    data = np.load(path)
    return data["best_theta"] if "best_theta" in data else data["theta"]


def cmd_info(args) -> int:
    from .agents.net import ConnectomeBrain, driving_subgraph
    from .connectome import build_surrogate
    from .sim.track import TRACKS
    from .sim import get_track

    conn = build_surrogate(scale=args.scale)
    drive = driving_subgraph(conn)
    brain = ConnectomeBrain(drive)
    print(conn.summary())
    print(drive.summary(), "  (mushroom body excluded)")
    print(f"free parameters: {brain.n_params}")
    print(f"readout: {len(brain.readout)} descending neurons")
    print(f"sensory ports: {', '.join(brain.input_ports)}")
    print("\npopulations:")
    for port in ("heading", "goal", "PFL3L", "PFL3R", "LAL_L", "turn_L", "turn_R", "speed", "stop"):
        if port in drive.ports:
            print(f"  {port:9s} {len(drive.port(port)):4d}")
    print("\ntracks:")
    for name in sorted(TRACKS):
        track = get_track(name)
        print(f"  {name:10s} {track.length:6.0f} m  min radius {1/np.abs(track.kappa).max():5.0f} m")
    return 0


def cmd_reference(args) -> int:
    from .agents.classical import ReferenceDriver
    from .sim import make_env

    env = make_env(args.track, n_envs=1, random_start=False, max_seconds=400.0)
    out = env.rollout(ReferenceDriver(half_width=env.track.half_width))
    lap = out["lap_time"][0]
    if np.isnan(lap):
        print(f"reference driver did not finish (retired code {out['retired'][0]})")
        return 1
    print(f"{args.track}: {lap:.2f} s over {env.track.length:.0f} m "
          f"({env.track.length / lap * 3.6:.1f} km/h average)")
    return 0


def cmd_probe(args) -> int:
    """Show the central complex turning heading error into a steering command."""
    from .agents.net import ConnectomeBrain, driving_subgraph
    from .connectome import apply_control, build_surrogate
    from .sim.obs import OBS_DIM, RING_SLICE

    conn = apply_control(build_surrogate(), args.control, seed=1)
    brain = ConnectomeBrain(driving_subgraph(conn))
    errors = np.linspace(-np.pi, np.pi, 25)
    heading = 0.7
    obs = np.zeros((len(errors), OBS_DIM))
    obs[:, RING_SLICE] = np.column_stack(
        [
            np.full(len(errors), np.sin(heading)),
            np.full(len(errors), np.cos(heading)),
            np.sin(heading - errors),
            np.cos(heading - errors),
        ]
    )
    brain.reset(len(errors))
    for _ in range(80):
        action = brain(obs)
    diff = brain.rates("PFL3R").mean(axis=0) - brain.rates("PFL3L").mean(axis=0)

    print(f"connectome control: {args.control}")
    print(" heading error   PFL3R-PFL3L    steer")
    for i in range(0, len(errors), 2):
        bar = int(abs(diff[i]) / (np.abs(diff).max() + 1e-9) * 24)
        side = ("-" * bar).rjust(24) if diff[i] < 0 else ("-" * bar).ljust(24)
        print(f"  {np.rad2deg(errors[i]):+7.0f} deg  {diff[i]:+8.4f} |{side}| {action[i,0]:+.3f}")
    print(f"\ncorr(PFL3 difference, sin(heading error)) = "
          f"{np.corrcoef(diff, np.sin(errors))[0,1]:+.3f}")
    print(f"corr(steer, heading error) = {np.corrcoef(action[:,0], errors)[0,1]:+.3f} "
          "(negative means corrective)")
    return 0


def cmd_train(args) -> int:
    from .learn.es import ESConfig
    from .learn.train import TrainConfig, evaluate_laps, train_curriculum

    cfg = TrainConfig(
        track=args.track,
        n_envs=args.envs,
        horizon_seconds=args.horizon,
        workers=args.workers,
        control=args.control,
        seed=args.seed,
        log_every=args.log_every,
        es=ESConfig(popsize=args.popsize, sigma=0.08, lr=0.05, seed=args.seed),
    )
    res = train_curriculum(
        cfg, imitation_generations=args.imitation, reward_generations=args.generations
    )
    print("\n", evaluate_laps(res["best_theta"], cfg))
    if args.out:
        np.savez(args.out, theta=res["theta"], best_theta=res["best_theta"])
        print(f"saved {args.out}")
    return 0


def cmd_evaluate(args) -> int:
    from .learn.train import TrainConfig, evaluate_laps

    cfg = TrainConfig(track=args.track, control=args.control, seed=args.seed)
    print(evaluate_laps(_load_theta(args.theta), cfg, track=args.track, n_envs=args.envs))
    return 0


def cmd_session(args) -> int:
    """Run repeated laps with mushroom-body plasticity on or off."""
    from .agents.classical import ReferenceDriver
    from .connectome import build_surrogate
    from .learn.mb import MBConfig, MushroomBody, run_learning_session
    from .sim import make_env

    conn = build_surrogate()
    env = make_env(
        args.track, n_envs=args.envs, random_start=False,
        max_seconds=args.laps * 90.0, target_laps=args.laps + 1,
    )
    if args.theta:
        from .agents.net import ConnectomeBrain, driving_subgraph

        policy = ConnectomeBrain(driving_subgraph(conn))
        policy.set_params(_load_theta(args.theta))
    else:
        policy = ReferenceDriver(half_width=env.track.half_width)
        print("no --theta given; using the reference driver as the base policy")

    for learn in (False, True):
        mb = MushroomBody(conn, MBConfig(n_sectors=args.sectors, seed=args.seed))
        out = run_learning_session(env, policy, mb, laps=args.laps, learn=learn, seed=args.seed)
        per = np.nanmean(out["lap_times"], axis=1)
        label = "plasticity ON " if learn else "plasticity OFF"
        print(f"\n{label}  laps={out['laps_completed']} surviving={out['alive'].sum()}/{args.envs}")
        print("  " + "  ".join(f"{t:.2f}" for t in per if not np.isnan(t)))
        if np.sum(~np.isnan(per)) >= 6:
            head, tail = np.nanmean(per[:3]), np.nanmean(per[-3:])
            print(f"  first 3 {head:.2f}s -> last 3 {tail:.2f}s  ({tail-head:+.2f}s)")
    return 0


def cmd_bridge(args) -> int:
    from .agents.net import ConnectomeBrain, driving_subgraph
    from .connectome import build_surrogate
    from .bridge.run import BridgeConfig, drive, record

    if args.action == "record":
        track = record(port=args.port, seconds=args.seconds, half_width=args.half_width)
        np.savez(args.track_file, xy=track.xy, half_width=track.half_width)
        print(f"saved {args.track_file}")
        return 0

    from .sim.track import Track

    data = np.load(args.track_file)
    track = Track.from_controls(data["xy"], half_width=float(data["half_width"]))
    brain = ConnectomeBrain(driving_subgraph(build_surrogate()))
    brain.set_params(_load_theta(args.theta))
    print(f"driving {track.length:.0f} m track. Ctrl-C to stop.")
    print(drive(brain, track, BridgeConfig(port=args.port, dry_run=args.dry_run)))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="flydrive", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("info", help="summarise the connectome and tracks")
    p.add_argument("--scale", type=float, default=1.0)
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("reference", help="lap time of the classical driver")
    p.add_argument("--track", default="national")
    p.set_defaults(func=cmd_reference)

    p = sub.add_parser("probe-cx", help="show the central-complex steering signal")
    p.add_argument("--control", default="none")
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("train", help="train a connectome brain to drive")
    p.add_argument("--track", default="national")
    p.add_argument("--generations", type=int, default=200)
    p.add_argument("--imitation", type=int, default=30)
    p.add_argument("--popsize", type=int, default=24)
    p.add_argument("--envs", type=int, default=24)
    p.add_argument("--horizon", type=float, default=18.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--control", default="none")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--out", default="")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="lap times for a saved parameter set")
    p.add_argument("theta")
    p.add_argument("--track", default="national")
    p.add_argument("--control", default="none")
    p.add_argument("--envs", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("session", help="multi-lap session with MB plasticity")
    p.add_argument("--theta", default="")
    p.add_argument("--track", default="national")
    p.add_argument("--laps", type=int, default=16)
    p.add_argument("--envs", type=int, default=24)
    p.add_argument("--sectors", type=int, default=24)
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(func=cmd_session)

    p = sub.add_parser("bridge", help="record a track or drive F1 25")
    p.add_argument("action", choices=["record", "drive"])
    p.add_argument("--port", type=int, default=20777)
    p.add_argument("--seconds", type=float, default=180.0)
    p.add_argument("--half-width", type=float, default=6.0)
    p.add_argument("--track-file", default="track.npz")
    p.add_argument("--theta", default="")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_bridge)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

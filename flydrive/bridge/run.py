"""Live bridge: drive F1 25 with a trained connectome brain.

    1. Record a centreline (one clean lap, driven by you, down the middle).
    2. Load a trained parameter set.
    3. Run the loop: telemetry in, observation built by the *same* code the
       simulator uses, brain, virtual gamepad out.

Everything the brain sees here it also saw in training, so what remains between
simulator and game is a vehicle-dynamics gap, not a representation gap. Expect
to need domain randomisation over grip and mass in training before this
transfers cleanly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..sim.obs import observe
from ..sim.track import Track
from ..sim.vehicle import VehicleParams
from .f1_udp import TelemetryReceiver, record_centreline
from .gamepad import open_gamepad


@dataclass
class BridgeConfig:
    port: int = 20777
    rate_hz: float = 50.0
    half_width: float = 6.0
    dry_run: bool = False
    max_seconds: float = 3600.0
    smoothing: float = 0.35      # low-pass on outgoing commands


def build_track_from_lap(points: np.ndarray, half_width: float = 6.0) -> Track:
    """Turn recorded positions into a Track with curvature and Frenet frame."""
    if len(points) < 16:
        raise ValueError(f"only {len(points)} centreline points; drive a full lap")
    return Track.from_controls(points, half_width=half_width, ds=1.0, name="recorded")


def drive(
    brain,
    track: Track,
    config: BridgeConfig | None = None,
    params: VehicleParams | None = None,
    on_step=None,
) -> dict:
    """Run the closed loop against the game until interrupted."""
    cfg = config or BridgeConfig()
    params = params or VehicleParams()
    receiver = TelemetryReceiver(port=cfg.port)
    pad = open_gamepad(dry_run=cfg.dry_run)

    brain.reset(1)
    last_action = np.zeros((1, 3))
    smoothed = np.zeros(3)
    hint = np.zeros(1, dtype=np.int64)
    period = 1.0 / cfg.rate_hz
    started = time.time()
    steps = 0

    try:
        while time.time() - started < cfg.max_seconds:
            tick = time.time()
            state = receiver.poll()
            if not state.valid:
                time.sleep(period)
                continue

            obs, idx, _s, e_y, e_psi, _goal = observe(
                track,
                np.array([state.x]),
                np.array([state.y]),
                np.array([state.heading]),
                np.array([state.vx]),
                np.array([state.vy]),
                np.array([state.yaw_rate]),
                last_action=last_action,
                hint=hint,
                params=params,
            )
            hint = idx
            action = brain(obs)[0]
            smoothed = (1 - cfg.smoothing) * smoothed + cfg.smoothing * action
            pad.send(*smoothed)
            last_action = smoothed[None, :].copy()
            steps += 1
            if on_step:
                on_step(state, obs, smoothed, float(e_y[0]), float(e_psi[0]))

            sleep = period - (time.time() - tick)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        pass
    finally:
        pad.release()
        receiver.close()
    return {"steps": steps, "seconds": time.time() - started}


def record(port: int = 20777, seconds: float = 180.0, half_width: float = 6.0) -> Track:
    """Interactive helper: drive a lap, get a Track back."""
    receiver = TelemetryReceiver(port=port)
    print(f"Listening on UDP {port}. Drive one clean lap down the centre of the track.")
    try:
        points = record_centreline(receiver, seconds=seconds)
    finally:
        receiver.close()
    track = build_track_from_lap(points, half_width=half_width)
    print(f"Recorded {len(points)} points -> track length {track.length:.0f} m")
    return track

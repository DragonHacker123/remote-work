"""Vectorised racing environment.

Runs many independent cars on one track in lockstep with NumPy, which is what
makes evolution strategies practical on a CPU: a population of 128 perturbed
brains is one batched rollout, not 128 sequential ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .obs import OBS_DIM, observe
from .track import Track, get_track
from .vehicle import VehicleParams, cornering_limit, step_dynamics


@dataclass
class EnvConfig:
    n_envs: int = 64
    control_hz: float = 50.0
    substeps: int = 4          # 200 Hz physics; tyre forces are stiff
    max_seconds: float = 200.0
    target_laps: float = 1.0
    random_start: bool = True
    start_speed: tuple[float, float] = (25.0, 55.0)
    # Metres past the white line before a car retires. Measured from the edge
    # rather than as a multiple of the room available, because on a racing line
    # the room on the inside of a hairpin is a few centimetres and scaling the
    # retirement threshold by it would delete the car for a normal apex.
    offtrack_slack: float = 4.8
    spin_limit: float = 1.5          # rad of heading error before retiring
    crash_penalty: float = 30.0
    progress_scale: float = 0.1
    offtrack_scale: float = 0.5
    jerk_scale: float = 0.05
    params: VehicleParams = field(default_factory=VehicleParams)


class RaceEnv:
    """A batch of cars on one circuit.

    ``step`` takes actions of shape (B, 3): steer in [-1, 1], throttle in
    [0, 1], brake in [0, 1]. Cars that retire are frozen and produce zero
    reward thereafter, so a fixed-horizon rollout is a valid fitness measure.
    """

    def __init__(self, track: Track | str = "spa", config: EnvConfig | None = None, seed: int = 0):
        self.track = get_track(track) if isinstance(track, str) else track
        self.cfg = config or EnvConfig()
        self.dt = 1.0 / self.cfg.control_hz
        self.rng = np.random.default_rng(seed)
        self.n = self.cfg.n_envs
        self.reset()

    # ----------------------------------------------------------------- reset

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        n, tr = self.n, self.track

        if self.cfg.random_start:
            start = self.rng.integers(0, tr.n, size=n)
        else:
            start = np.zeros(n, dtype=np.int64)

        # Start at a speed the local corner can actually hold, otherwise half
        # the population retires in the first second and carries no signal.
        kappa = np.abs(tr.kappa[start]) + 1e-4
        lo, hi = self.cfg.start_speed
        v_corner = np.sqrt(cornering_limit(np.full(n, hi), self.cfg.params) / kappa)
        v0 = np.clip(v_corner, lo, hi) * self.rng.uniform(0.6, 0.95, n)

        psi0 = tr.psi[start] + self.rng.normal(0.0, 0.04, n)
        # Scatter across the road, but by the room actually available on each
        # side: on a racing line the apex of a hairpin has almost none.
        frac = self.rng.uniform(-0.3, 0.3, n)
        lat = frac * np.where(frac >= 0.0, tr.hw_left[start], tr.hw_right[start])
        x0 = tr.xy[start, 0] - np.sin(tr.psi[start]) * lat
        y0 = tr.xy[start, 1] + np.cos(tr.psi[start]) * lat

        self.state = np.stack(
            [x0, y0, psi0, v0, np.zeros(n), np.zeros(n)], axis=1
        ).astype(np.float64)
        self.idx = start.astype(np.int64)
        self.last_action = np.zeros((n, 3))
        self.alpha_f = np.zeros(n)
        self.alpha_r = np.zeros(n)
        self.alive = np.ones(n, dtype=bool)
        self.time = np.zeros(n)
        self.progress = np.zeros(n)
        self.lap_time = np.full(n, np.nan)
        self.retired_reason = np.zeros(n, dtype=np.int8)  # 0 none 1 off 2 spin 3 stall
        _, s, _, _ = tr.project(self.state[:, 0], self.state[:, 1], self.idx)
        self.prev_s = s
        return self._observe()

    # ------------------------------------------------------------- observing

    def _observe(self) -> np.ndarray:
        x, y, psi, vx, vy, r = (self.state[:, i] for i in range(6))
        obs, idx, s, e_y, e_psi, goal = observe(
            self.track,
            x, y, psi, vx, vy, r,
            last_action=self.last_action,
            hint=self.idx,
            params=self.cfg.params,
            slip=(self.alpha_f, self.alpha_r),
        )
        self.idx, self.cur_s, self.e_y, self.e_psi = idx, s, e_y, e_psi
        self.goal_angle = goal
        return obs

    # ----------------------------------------------------------------- step

    def step(self, action: np.ndarray):
        cfg, tr = self.cfg, self.track
        action = np.asarray(action, dtype=np.float64)
        steer = np.clip(action[:, 0], -1.0, 1.0)
        throttle = np.clip(action[:, 1], 0.0, 1.0)
        brake = np.clip(action[:, 2], 0.0, 1.0)

        # Rate-limit the steering the way a real steering rack does; without
        # it a policy can chatter the wheel to fake grip it does not have.
        max_delta = cfg.params.steer_rate / cfg.params.max_steer * self.dt
        steer = np.clip(steer, self.last_action[:, 0] - max_delta, self.last_action[:, 0] + max_delta)

        live = self.alive
        sub_dt = self.dt / cfg.substeps
        grade = tr.grade[self.idx]
        vcurv = tr.vcurv[self.idx]
        for _ in range(cfg.substeps):
            new_state, af, ar = step_dynamics(
                self.state, steer, throttle, brake, cfg.params, sub_dt,
                grade=grade, vcurv=vcurv,
            )
            self.state = np.where(live[:, None], new_state, self.state)
            self.alpha_f = np.where(live, af, self.alpha_f)
            self.alpha_r = np.where(live, ar, self.alpha_r)

        self.time += np.where(live, self.dt, 0.0)
        self.last_action = np.stack([steer, throttle, brake], axis=1)

        obs = self._observe()

        # Progress along the centreline, handling the start/finish wrap.
        length = tr.length
        delta = (self.cur_s - self.prev_s + length / 2) % length - length / 2
        delta = np.where(live, delta, 0.0)
        self.prev_s = self.cur_s
        before = self.progress
        self.progress = self.progress + delta

        reward = cfg.progress_scale * delta
        # Room on the side the car has drifted towards. On the centreline the
        # two are equal and this is the old symmetric test; on a racing line
        # they are not, and using the wrong one puts the edge in the wrong place.
        room = np.where(self.e_y >= 0.0, tr.hw_left[self.idx], tr.hw_right[self.idx])
        beyond = np.maximum(np.abs(self.e_y) - room, 0.0)
        over = beyond / tr.hw[self.idx]
        reward -= cfg.offtrack_scale * over * over
        reward -= cfg.jerk_scale * np.abs(steer - action[:, 0])
        reward = np.where(live, reward, 0.0)

        # Lap completion.
        target = length * cfg.target_laps
        finished = live & (before < target) & (self.progress >= target)
        self.lap_time = np.where(finished & np.isnan(self.lap_time), self.time, self.lap_time)

        off = beyond > cfg.offtrack_slack
        spun = np.abs(self.e_psi) > cfg.spin_limit
        stalled = (self.state[:, 3] < 2.0) & (self.time > 3.0)
        timeout = self.time > cfg.max_seconds
        bad = live & (off | spun | stalled)
        reward -= cfg.crash_penalty * bad
        self.retired_reason = np.where(
            bad & (self.retired_reason == 0),
            np.where(off, 1, np.where(spun, 2, 3)),
            self.retired_reason,
        )

        done = bad | (live & (timeout | finished))
        self.alive = live & ~done
        info = {
            "progress": self.progress,
            "lap_time": self.lap_time,
            "alive": self.alive,
            "retired": self.retired_reason,
            "speed": self.state[:, 3],
            "e_y": self.e_y,
        }
        return obs, reward, done, info

    # ------------------------------------------------------------- rollouts

    def rollout(
        self,
        policy,
        horizon: int | None = None,
        record: bool = False,
        seed: int | None = None,
    ):
        """Run until every car retires or ``horizon`` control steps elapse.

        ``policy`` maps (B, OBS_DIM) observations to (B, 3) actions and may
        carry recurrent state; it is reset via ``policy.reset(B)`` if present.
        """
        if horizon is None:
            horizon = int(self.cfg.max_seconds * self.cfg.control_hz)
        obs = self.reset(seed=seed)
        if hasattr(policy, "reset"):
            policy.reset(self.n)
        total = np.zeros(self.n)
        trace = [] if record else None
        for _ in range(horizon):
            action = policy(obs)
            obs, reward, _done, info = self.step(action)
            total += reward
            if record:
                trace.append(np.column_stack([self.state[:, :2], self.state[:, 3], action]))
            if not self.alive.any():
                break
        out = {
            "return": total,
            "progress": self.progress.copy(),
            "lap_time": self.lap_time.copy(),
            "retired": self.retired_reason.copy(),
            "time": self.time.copy(),
        }
        if record:
            out["trace"] = np.array(trace)
        return out


def make_env(track: str = "spa", n_envs: int = 64, seed: int = 0, **kw) -> RaceEnv:
    cfg = EnvConfig(n_envs=n_envs, **kw)
    return RaceEnv(track=track, config=cfg, seed=seed)


__all__ = ["RaceEnv", "EnvConfig", "make_env", "OBS_DIM"]

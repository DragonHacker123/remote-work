"""Mushroom-body plasticity: the within-session learning loop.

The ES loop in ``learn.es`` is phylogeny -- it tunes what the animal is born
with, over many lifetimes. This is ontogeny: a single car, on a single track,
getting faster lap after lap with its inherited parameters untouched. That is
what "learn to improve its lap times" actually asks for, and the fly already
has the circuit for it.

The mechanism follows the real mushroom body:

  * Projection neurons carry the context. In a fly that is odour; here it is
    where the car is on the circuit, how fast, and how tight the corner is.
  * Kenyon cells expand that into a high-dimensional code, each sampling a
    handful of PNs at random. The connectome's own PN->KC wiring is used.
  * APL, a single giant inhibitory neuron, normalises the population so only
    the most strongly driven KCs fire. That is what makes the code *sparse*,
    and sparseness is what keeps one corner's memory from overwriting the next
    corner's.
  * Dopaminergic neurons report whether things went better or worse than
    expected, and gate plasticity at the KC->MBON synapse.
  * MBON output biases the motor command.

The update is the three-factor rule the MB is built around: presynaptic KC
activity, times the postsynaptic perturbation, times dopamine. Canonically DANs
*depress* the KC->MBON synapses active during punishment; signed
reward-modulated Hebbian learning is the same rule with both signs allowed, and
it is what makes the circuit a working reinforcement learner rather than only
an avoidance one.

Reward is each sector's time against the best the car has managed there so far,
so the baseline is the driver's own history -- a dopaminergic prediction error,
not an absolute score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

from ..connectome.schema import Connectome


@dataclass
class MBConfig:
    n_sectors: int = 24           # how finely the circuit is carved up
    sparsity: int = 10            # KCs allowed to fire per context (APL gain control)
    lr: float = 0.15
    explore: float = 0.035        # exploratory perturbation of the MBON output
    weight_clip: float = 1.0
    # The MBON output is a *trim* on an already-competent driver, not a
    # controller. Left unbounded it accumulates across sector visits until the
    # steering bias alone puts the car in the wall.
    trim_scale: float = 0.12
    reward_scale: float = 4.0     # seconds^-1, converts sector-time delta to dopamine
    place_width: float = 1.4      # sectors, width of each PN's place field
    seed: int = 0


class MushroomBody:
    """Sparse-coding associative memory over track position.

    Outputs a two-element motor bias: a steering trim and a speed trim, applied
    on top of whatever the inherited network already does.
    """

    N_OUTPUTS = 2                 # steer trim, speed trim

    def __init__(self, conn: Connectome, config: MBConfig | None = None):
        self.cfg = config or MBConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

        pn = conn.port("context")
        kc = conn.port("KC")
        self.n_pn, self.n_kc = len(pn), len(kc)

        # Pull the connectome's real PN->KC block out as a dense matrix. This
        # is the random expansion that makes contexts linearly separable, and
        # it is inherited rather than learned.
        mask = np.isin(conn.pre, pn) & np.isin(conn.post, kc)
        pn_pos = {int(v): i for i, v in enumerate(pn)}
        kc_pos = {int(v): i for i, v in enumerate(kc)}
        rows = np.array([kc_pos[int(v)] for v in conn.post[mask]], dtype=np.int32)
        cols = np.array([pn_pos[int(v)] for v in conn.pre[mask]], dtype=np.int32)
        self.pn_to_kc = csr_matrix(
            (conn.weight[mask].astype(np.float64), (rows, cols)),
            shape=(self.n_kc, self.n_pn),
        )

        # Each PN is a place cell for one stretch of track.
        self.pn_centre = np.linspace(0.0, self.cfg.n_sectors, self.n_pn, endpoint=False)
        self.weights = np.zeros((self.n_kc, self.N_OUTPUTS))
        self.best_sector_time: np.ndarray | None = None
        self.reset_memory()

    # ------------------------------------------------------------------ code

    def reset_memory(self) -> None:
        self.weights[:] = 0.0
        self.best_sector_time = None

    def pn_activity(self, sector_pos: np.ndarray) -> np.ndarray:
        """Place-field activity over PNs for a fractional sector position (B,)."""
        n = self.cfg.n_sectors
        d = (sector_pos[None, :] - self.pn_centre[:, None] + n / 2) % n - n / 2
        return np.exp(-0.5 * (d / self.cfg.place_width) ** 2)

    def kc_code(self, sector_pos: np.ndarray) -> np.ndarray:
        """Sparse binary Kenyon-cell code, (n_kc, B).

        APL's job in one line: keep only the top ``sparsity`` most strongly
        driven cells. Without it every context recruits every KC, the codes
        stop being separable, and learning one corner unlearns the others.
        """
        drive = self.pn_to_kc @ self.pn_activity(sector_pos)
        k = self.cfg.sparsity
        if k >= self.n_kc:
            return (drive > 0).astype(np.float64)
        cut = np.partition(drive, -k, axis=0)[-k, :]
        return (drive >= cut[None, :]).astype(np.float64)

    # --------------------------------------------------------------- readout

    def output(self, code: np.ndarray) -> np.ndarray:
        """MBON output (B, 2) for a KC code.

        Divided by the number of active Kenyon cells, so the trim is a mean
        synaptic weight rather than a sum that grows with code density, and
        scaled into the small range a trim should occupy.
        """
        active = np.maximum(code.sum(axis=0), 1.0)
        return (code.T @ self.weights) / active[:, None] * self.cfg.trim_scale

    def perturb(self, batch: int) -> np.ndarray:
        """Exploratory noise on the MBON output -- the thing dopamine judges."""
        return self.rng.normal(0.0, self.cfg.explore, (batch, self.N_OUTPUTS))

    # -------------------------------------------------------------- learning

    def update(self, code: np.ndarray, perturbation: np.ndarray, reward: np.ndarray) -> None:
        """Three-factor update: KC activity x perturbation x dopamine.

        ``code`` is (n_kc, B), ``perturbation`` and ``reward`` are (B, 2) and
        (B,). Only the KCs that actually fired for this context are eligible,
        which is what keeps the memory local to the corner it was formed in.
        """
        signal = perturbation * reward[:, None] * self.cfg.lr
        self.weights += code @ signal
        np.clip(self.weights, -self.cfg.weight_clip, self.cfg.weight_clip, out=self.weights)

    def dopamine(self, sector: np.ndarray, elapsed: np.ndarray) -> np.ndarray:
        """Reward prediction error: this sector's time against the best so far."""
        if self.best_sector_time is None:
            self.best_sector_time = np.full(self.cfg.n_sectors, np.nan)
        best = self.best_sector_time[sector]
        reward = np.where(np.isnan(best), 0.0, (best - elapsed) * self.cfg.reward_scale)
        improved = np.isnan(best) | (elapsed < best)
        # Update the baseline where the car beat it, per sector.
        for s, t, ok in zip(sector, elapsed, improved):
            if ok:
                self.best_sector_time[s] = t
        return np.clip(reward, -3.0, 3.0)


def run_learning_session(
    env,
    brain,
    mb: MushroomBody,
    laps: int = 20,
    learn: bool = True,
    seed: int = 0,
) -> dict:
    """Drive repeated laps, letting the mushroom body tune the motor bias.

    The inherited parameters of ``brain`` are never touched. Everything that
    improves across laps lives in the KC->MBON weights.

    Caveat worth stating plainly: every car in the batch shares one mushroom
    body, so ``n_envs`` cars contribute ``n_envs`` independent perturbations per
    sector per lap. That is variance reduction, not biology -- a single fly has
    one brain and would need proportionally more laps to extract the same
    signal. The learning rule is identical either way; only the sample rate
    differs.
    """
    track = env.track
    n = env.n
    if env.cfg.target_laps < laps:
        raise ValueError(
            f"env retires cars after {env.cfg.target_laps} lap(s) but this session "
            f"wants {laps}. Build the env with target_laps >= laps (and a "
            "max_seconds budget to match)."
        )
    obs = env.reset(seed=seed)
    brain.reset(n)

    def sector_of(idx: np.ndarray) -> np.ndarray:
        return (idx * mb.cfg.n_sectors // track.n).astype(np.int64)

    cur_sector = sector_of(env.idx)
    sector_start = np.zeros(n)
    active_code = mb.kc_code(cur_sector.astype(float))
    active_perturb = mb.perturb(n) if learn else np.zeros((n, mb.N_OUTPUTS))

    # One slot per car per lap; cars cross the line at different times, so a
    # shared row per crossing would scramble whose lap was whose.
    stamps = np.full((laps, n), np.nan)
    lap_index = np.zeros(n, dtype=np.int64)
    max_steps = int(env.cfg.max_seconds / env.dt)

    for _ in range(max_steps):
        pos = env.idx * mb.cfg.n_sectors / track.n
        code = mb.kc_code(pos)
        bias = mb.output(code) + active_perturb

        action = brain(obs)
        action[:, 0] = np.clip(action[:, 0] + bias[:, 0], -1.0, 1.0)
        action[:, 1] = np.clip(action[:, 1] + bias[:, 1], 0.0, 1.0)
        action[:, 2] = np.clip(action[:, 2] - 0.5 * bias[:, 1], 0.0, 1.0)
        obs, _r, _done, _info = env.step(action)

        new_sector = sector_of(env.idx)
        crossed = (new_sector != cur_sector) & env.alive
        if crossed.any():
            idx = np.flatnonzero(crossed)
            elapsed = env.time[idx] - sector_start[idx]
            if learn:
                reward = mb.dopamine(cur_sector[idx], elapsed)
                mb.update(active_code[:, idx], active_perturb[idx], reward)
            sector_start[idx] = env.time[idx]
            cur_sector[idx] = new_sector[idx]
            fresh = mb.perturb(len(idx)) if learn else np.zeros((len(idx), mb.N_OUTPUTS))
            active_perturb[idx] = fresh
            active_code[:, idx] = code[:, idx]

        crossed_line = env.alive & (lap_index < laps)
        crossed_line &= env.progress >= (lap_index + 1) * track.length
        if crossed_line.any():
            cars = np.flatnonzero(crossed_line)
            stamps[lap_index[cars], cars] = env.time[cars]
            lap_index[cars] += 1
        if not env.alive.any() or lap_index.min() >= laps:
            break

    # Crossing times are cumulative; differencing down the lap axis per car
    # turns them into individual lap times.
    per_lap = np.diff(np.vstack([np.zeros((1, n)), stamps]), axis=0)
    return {
        "lap_times": per_lap,
        "laps_completed": int(lap_index.max()) if n else 0,
        "alive": env.alive.copy(),
        "weights_norm": float(np.abs(mb.weights).sum()),
    }

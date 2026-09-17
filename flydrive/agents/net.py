"""The connectome-constrained brain.

A firing-rate network whose connectivity is *fixed* by the connectome. What
gets learned is deliberately low-dimensional and biologically shaped:

  * one output gain per cell type -- a genome encodes cell-type-level rules,
    not fifty million individual synaptic weights. Gains are constrained
    positive, because a cell type cannot flip its neurotransmitter to suit the
    optimiser.
  * one membrane time constant and one resting drive per cell type.
  * a sensory encoder, since a fly has no speedometer and we have to invent
    the transduction.
  * a readout from the descending neurons, initialised to the fly's own motor
    convention: DNa02 left-right difference steers, DNa01 sets forward speed,
    DNp09 stops.

Heading and goal are *not* encoded as a precomputed error. They go in as two
separate bumps of activity on the EPG and FC2 rings, and the central complex
computes the difference itself through the PFL3 shift -- that circuit is the
reason this project is worth doing, so handing it the answer would be cheating.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..connectome.schema import Connectome
from ..sim.obs import FEATURE_SLICE, OBS_DIM, ring_angles

FEATURE_DIM = FEATURE_SLICE.stop - FEATURE_SLICE.start

# Resting drive for the central-complex output populations. See the note in
# ``initial_params`` -- this is the single most important initialisation
# constant in the model.
THRESHOLD_BIAS = -1.0


@dataclass
class BrainConfig:
    control_dt: float = 0.02
    inner_steps: int = 2          # neural steps per control step
    tau_min: float = 0.015        # s; must exceed the neural step for stability
    tau_max: float = 0.300
    rate_max: float = 5.0         # saturating firing rate, arbitrary units
    enc_channels: int = 24
    input_ports: tuple[str, ...] = ("T4T5", "LPLC2", "AN", "context")
    readout_ports: tuple[str, ...] = ("turn_L", "turn_R", "speed", "stop")
    ring_ports: tuple[str, ...] = ("heading", "goal")
    seed: int = 0


class _ParamLayout:
    """Names contiguous slices of a flat parameter vector."""

    def __init__(self) -> None:
        self.slices: dict[str, slice] = {}
        self.shapes: dict[str, tuple[int, ...]] = {}
        self.size = 0

    def add(self, name: str, shape: tuple[int, ...]) -> None:
        n = int(np.prod(shape))
        self.slices[name] = slice(self.size, self.size + n)
        self.shapes[name] = shape
        self.size += n

    def unpack(self, theta: np.ndarray) -> dict[str, np.ndarray]:
        return {k: theta[s].reshape(self.shapes[k]) for k, s in self.slices.items()}


class ConnectomeBrain:
    """Rate network over a fixed connectome, driving the car through its DNs."""

    def __init__(self, conn: Connectome, config: BrainConfig | None = None):
        self.conn = conn
        self.cfg = config or BrainConfig()
        rng = np.random.default_rng(self.cfg.seed)

        # Drop the KC->MBON block: those weights belong to the plasticity
        # module, which adds its contribution as an overlay.
        exclude = []
        if "KC" in conn.type_names and "MBON" in conn.type_names:
            exclude.append(("KC", "MBON"))
        self.op = conn.build_operator(exclude_pairs=exclude)
        self.n = conn.n_neurons
        self.n_types = conn.n_types

        self.input_ports = [p for p in self.cfg.input_ports if p in conn.ports]
        self.ring_ports = [p for p in self.cfg.ring_ports if p in conn.ports]
        self.readout = np.concatenate(
            [conn.port(p) for p in self.cfg.readout_ports if p in conn.ports]
        )

        # Fixed random expansion from encoder channels to sensory neurons.
        # The learned part stays small; diversity of tuning comes from here,
        # which is also how a real sensory population gets it.
        self.expand = {
            p: rng.normal(0.0, 1.0, (self.cfg.enc_channels, len(conn.port(p))))
            / np.sqrt(self.cfg.enc_channels)
            for p in self.input_ports
        }
        # Preferred angle of each wedge on the CX rings.
        self.ring_angles_of = {
            p: np.linspace(0.0, 2.0 * np.pi, len(conn.port(p)), endpoint=False)
            for p in self.ring_ports
        }

        self.layout = _ParamLayout()
        self.layout.add("log_gain", (self.n_types,))
        self.layout.add("log_tau", (self.n_types,))
        self.layout.add("bias", (self.n_types,))
        for p in self.input_ports:
            self.layout.add(f"enc_{p}", (FEATURE_DIM, self.cfg.enc_channels))
        for p in self.ring_ports:
            self.layout.add(f"ring_{p}", (2,))          # log amplitude, log width
        # Two output channels, not three: steering, and one *signed*
        # longitudinal command. Independent throttle and brake channels let the
        # network learn a state where it neither accelerates nor coasts but
        # holds a light brake -- which is exactly what a distilled readout does
        # off-distribution, and it is self-reinforcing, because braking takes
        # the car further from any state the teacher demonstrated.
        self.layout.add("dec_w", (len(self.readout), 2))
        self.layout.add("dec_b", (2,))

        self.theta = self.initial_params()
        self.set_params(self.theta)
        self.state = np.zeros((self.n, 1))

    # ------------------------------------------------------------ parameters

    @property
    def n_params(self) -> int:
        return self.layout.size

    def initial_params(self) -> np.ndarray:
        """Start from unit gains and the fly's own motor convention."""
        theta = np.zeros(self.layout.size)
        p = self.layout.slices
        theta[p["log_tau"]] = np.log(0.025)   # faster than 40 ms; steering must not lag

        bias = np.full(self.n_types, 0.05)
        # The PFL3 populations must sit *near threshold*. Their steering signal
        # is the difference between two rectified population sums, and a sum of
        # rectified units only depends on where the bumps are while some units
        # are actually clipped at zero. Let every wedge float above threshold
        # and the sum becomes linear -- at which point the difference between
        # the two bumps is a constant and the steering signal vanishes entirely.
        # Measured on the surrogate: threshold bias takes the slope of
        # (PFL3R - PFL3L) with respect to heading error from 0.0007 to 0.32.
        for name in ("PFL3L", "PFL3R", "PFL2"):
            if name in self.conn.type_names:
                bias[self.conn.type_index(name)] = THRESHOLD_BIAS
        theta[p["bias"]] = bias
        # Seeded by position, not by hash(port): Python string hashing is
        # randomised per process and would make runs unreproducible.
        for i, port in enumerate(self.input_ports):
            theta[p[f"enc_{port}"]] = (
                np.random.default_rng(self.cfg.seed * 1009 + i)
                .normal(0.0, 0.3, self.layout.shapes[f"enc_{port}"])
                .ravel()
            )
        # Drive the rings hard enough that the PFL3 populations rectify: if
        # every wedge sits well above threshold the left-right sum degenerates
        # to a constant and the steering signal vanishes.
        for port in self.ring_ports:
            theta[p[f"ring_{port}"]] = [np.log(3.0), np.log(4.0)]

        # Biological readout: steering is the left-right difference across
        # DNa02, forward drive comes from DNa01, braking from DNp09.
        dec = np.zeros((len(self.readout), 2))
        offset = 0
        for name in self.cfg.readout_ports:
            if name not in self.conn.ports:
                continue
            size = len(self.conn.port(name))
            sl = slice(offset, offset + size)
            if name == "turn_L":
                dec[sl, 0] = +1.0 / size
            elif name == "turn_R":
                dec[sl, 0] = -1.0 / size
            elif name == "speed":
                dec[sl, 1] = +1.0 / size     # DNa01 drives forward
            elif name == "stop":
                dec[sl, 1] = -1.0 / size     # DNp09 opposes it
            offset += size
        theta[p["dec_w"]] = dec.ravel()
        theta[p["dec_b"]] = [0.0, 0.3]
        return theta

    def set_params(self, theta: np.ndarray) -> None:
        self.theta = np.asarray(theta, dtype=np.float64)
        self.par = self.layout.unpack(self.theta)
        types = self.conn.type_ids

        # Gains are strictly positive: the connectome already carries the sign.
        self.op.set_gains(np.exp(np.clip(self.par["log_gain"], -4.0, 4.0)))
        self.matrix = self.op.matrix

        tau = np.clip(
            np.exp(np.clip(self.par["log_tau"], -6.0, 2.0)),
            self.cfg.tau_min,
            self.cfg.tau_max,
        )
        dt_neural = self.cfg.control_dt / self.cfg.inner_steps
        self.alpha = (dt_neural / tau[types])[:, None]
        self.bias_n = self.par["bias"][types][:, None]

    # ---------------------------------------------------------------- inputs

    def _external(self, obs: np.ndarray) -> np.ndarray:
        """Build the external drive (N, B) from an observation batch."""
        b = obs.shape[0]
        u = np.zeros((self.n, b))
        feats = obs[:, FEATURE_SLICE]                                 # (B, F)
        for port in self.input_ports:
            channels = feats @ self.par[f"enc_{port}"]                # (B, C)
            u[self.conn.port(port)] = (channels @ self.expand[port]).T

        heading, goal = ring_angles(obs)
        for port, angle in zip(self.ring_ports, (heading, goal)):
            amp, width = np.exp(np.clip(self.par[f"ring_{port}"], -4.0, 4.0))
            theta = self.ring_angles_of[port][:, None]                # (W, 1)
            # von Mises bump: a single localised hill of activity, which is
            # what an EPG compass bump actually looks like.
            u[self.conn.port(port)] = amp * np.exp(
                width * (np.cos(theta - angle[None, :]) - 1.0)
            )
        return u

    # ----------------------------------------------------------------- loop

    def reset(self, n: int) -> None:
        self.state = np.zeros((self.n, n))

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        if self.state.shape[1] != obs.shape[0]:
            self.reset(obs.shape[0])
        u = self._external(obs)
        x = self.state
        for _ in range(self.cfg.inner_steps):
            r = np.clip(x, 0.0, self.cfg.rate_max)
            x = x + self.alpha * (-x + self.matrix @ r + self.bias_n + u)
            x = np.clip(x, -20.0, 20.0)
        self.state = x

        rates = np.clip(x[self.readout], 0.0, self.cfg.rate_max)      # (R, B)
        z = rates.T @ self.par["dec_w"] + self.par["dec_b"]           # (B, 2)
        longitudinal = z[:, 1]
        return np.column_stack(
            [
                np.tanh(z[:, 0]),
                np.clip(longitudinal, 0.0, 1.0),      # throttle
                np.clip(-longitudinal, 0.0, 1.0),     # brake
            ]
        )

    # ------------------------------------------------------------- utilities

    def rates(self, port: str) -> np.ndarray:
        """Current firing rates of a named population, for analysis."""
        return np.clip(self.state[self.conn.port(port)], 0.0, self.cfg.rate_max)

    def readout_rates(self) -> np.ndarray:
        """Descending-neuron rates as (B, R), the features the decoder reads."""
        return np.clip(self.state[self.readout], 0.0, self.cfg.rate_max).T

    def advance(self, obs: np.ndarray) -> None:
        """Step the network without producing an action (used for teacher forcing)."""
        self.__call__(obs)

    def set_readout(self, weights: np.ndarray, bias: np.ndarray) -> None:
        """Overwrite the decoder, e.g. from a least-squares fit."""
        theta = self.theta.copy()
        theta[self.layout.slices["dec_w"]] = weights.ravel()
        theta[self.layout.slices["dec_b"]] = bias.ravel()
        self.set_params(theta)

    def clone(self) -> "ConnectomeBrain":
        twin = ConnectomeBrain(self.conn, self.cfg)
        twin.set_params(self.theta.copy())
        return twin


def driving_subgraph(conn: Connectome) -> Connectome:
    """The sensorimotor loop without the mushroom body."""
    regions = tuple(r for r in conn.region_names if r != "MB")
    return conn.subgraph(regions=regions)

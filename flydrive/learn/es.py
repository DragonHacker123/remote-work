"""Evolution strategies for the outer loop.

ES rather than a policy-gradient method, for three reasons that all matter
here: there is no gradient to take through a sparse recurrent rate network
without building an autodiff graph over it; the parameter vector is small
(hundreds, not millions) because the connectome supplies the structure; and
fitness evaluations parallelise perfectly across cores.

Biologically this is the phylogenetic loop -- it tunes what the animal is born
with. The ontogenetic loop, which learns *this* circuit on *this* track within
a session, is the mushroom body in ``learn.mb``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ESConfig:
    popsize: int = 32             # rounded up to even; antithetic pairs
    sigma: float = 0.06
    lr: float = 0.04
    weight_decay: float = 0.003   # pulls back toward the starting point
    sigma_decay: float = 0.997
    sigma_min: float = 0.015
    beta1: float = 0.9
    beta2: float = 0.999
    seed: int = 0


def centered_ranks(fitness: np.ndarray) -> np.ndarray:
    """Map fitness to evenly spaced ranks in [-0.5, 0.5].

    Rank shaping makes the update invariant to the scale of the reward and
    immune to a single lucky rollout dominating the gradient -- which matters
    a lot when one car happens to survive a corner the rest spin at.
    """
    ranks = np.empty(len(fitness), dtype=np.float64)
    ranks[np.argsort(fitness)] = np.arange(len(fitness))
    return ranks / max(len(fitness) - 1, 1) - 0.5


class OpenAIES:
    """Antithetic-sampling ES with rank shaping and an Adam update."""

    def __init__(self, theta0: np.ndarray, config: ESConfig | None = None):
        self.cfg = config or ESConfig()
        self.theta = np.asarray(theta0, dtype=np.float64).copy()
        self.anchor = self.theta.copy()
        self.sigma = self.cfg.sigma
        self.rng = np.random.default_rng(self.cfg.seed)
        self._m = np.zeros_like(self.theta)
        self._v = np.zeros_like(self.theta)
        self._t = 0
        self._eps: np.ndarray | None = None
        self.history: list[dict] = []

    @property
    def popsize(self) -> int:
        return self.cfg.popsize + (self.cfg.popsize % 2)

    def ask(self) -> np.ndarray:
        """Return (popsize, n_params) candidate parameter vectors."""
        half = self.popsize // 2
        eps = self.rng.normal(size=(half, self.theta.size))
        self._eps = np.concatenate([eps, -eps], axis=0)
        return self.theta[None, :] + self.sigma * self._eps

    def tell(self, fitness: np.ndarray) -> None:
        if self._eps is None:
            raise RuntimeError("tell() called before ask()")
        fitness = np.asarray(fitness, dtype=np.float64)
        if fitness.shape != (self.popsize,):
            raise ValueError(f"expected {self.popsize} fitnesses, got {fitness.shape}")

        shaped = centered_ranks(fitness)
        grad = (self._eps.T @ shaped) / (self.popsize * self.sigma)
        # Decay toward the biologically initialised parameters rather than
        # toward zero: zero is not a neutral point for a log-gain.
        grad = grad - self.cfg.weight_decay * (self.theta - self.anchor)

        self._t += 1
        c = self.cfg
        self._m = c.beta1 * self._m + (1 - c.beta1) * grad
        self._v = c.beta2 * self._v + (1 - c.beta2) * grad * grad
        m_hat = self._m / (1 - c.beta1**self._t)
        v_hat = self._v / (1 - c.beta2**self._t)
        self.theta = self.theta + c.lr * m_hat / (np.sqrt(v_hat) + 1e-8)

        self.sigma = max(self.sigma * c.sigma_decay, c.sigma_min)
        self.history.append(
            {
                "gen": self._t,
                "mean": float(fitness.mean()),
                "max": float(fitness.max()),
                "sigma": float(self.sigma),
            }
        )
        self._eps = None

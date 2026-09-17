"""Null models for the connectome.

This is the part that decides whether the project is science or set dressing.
If a network built on real fly wiring does not beat a null model that matches
its degree sequence and cell-type composition, then the connectome contributed
nothing and what you have is an ordinary sparse recurrent network with an
unusually good origin story. Always run at least ``shuffle_within_type``.

Ordered from weakest to strongest control:

``randomize_weights``   topology kept, synapse counts permuted.
``shuffle_signs``       topology and weights kept, E/I identity permuted
                        across cell types.
``shuffle_pairing``     configuration model. Every neuron keeps its exact in-
                        and out-degree; which neuron talks to which is
                        destroyed, and so is all type-level structure.
``shuffle_within_type``  every neuron keeps its exact in- and out-degree *and*
                        the full cell-type-by-cell-type connectivity matrix is
                        preserved. Only neuron-level wiring specificity is
                        destroyed. This is the one that hurts.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .schema import Connectome


def _permute_within(
    key: np.ndarray, post: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Permute ``post`` independently within each block of equal ``key``."""
    base = np.argsort(key, kind="stable")
    order = np.lexsort((rng.random(key.size), key))
    out = post.copy()
    out[base] = post[order]
    return out


def shuffle_pairing(conn: Connectome, seed: int = 0) -> Connectome:
    """Configuration model. Exact degree preservation, no other structure."""
    rng = np.random.default_rng(seed)
    return replace(
        conn, post=rng.permutation(conn.post), name=f"{conn.name}+shuffled-pairing"
    )


def shuffle_within_type(conn: Connectome, seed: int = 0) -> Connectome:
    """Preserve degrees and the type-level connectivity matrix exactly."""
    rng = np.random.default_rng(seed)
    key = conn.type_ids[conn.pre].astype(np.int64) * conn.n_types + conn.type_ids[conn.post]
    return replace(
        conn,
        post=_permute_within(key, conn.post, rng),
        name=f"{conn.name}+shuffled-within-type",
    )


def shuffle_signs(conn: Connectome, seed: int = 0) -> Connectome:
    """Permute the excitatory/inhibitory assignment across cell types."""
    rng = np.random.default_rng(seed)
    # One sign per presynaptic type, then permuted between types.
    per_type = np.zeros(conn.n_types, dtype=np.int8)
    for t in range(conn.n_types):
        edges = conn.sign[conn.type_ids[conn.pre] == t]
        if edges.size:
            per_type[t] = np.sign(edges.sum()) or 1
    per_type = rng.permutation(per_type)
    return replace(
        conn, sign=per_type[conn.type_ids[conn.pre]], name=f"{conn.name}+shuffled-signs"
    )


def randomize_weights(conn: Connectome, seed: int = 0) -> Connectome:
    """Keep topology, permute synapse counts across edges."""
    rng = np.random.default_rng(seed)
    return replace(
        conn, weight=rng.permutation(conn.weight), name=f"{conn.name}+shuffled-weights"
    )


CONTROLS = {
    "none": lambda c, seed=0: c,
    "weights": randomize_weights,
    "signs": shuffle_signs,
    "pairing": shuffle_pairing,
    "within-type": shuffle_within_type,
}


def apply_control(conn: Connectome, kind: str, seed: int = 0) -> Connectome:
    if kind not in CONTROLS:
        raise KeyError(f"unknown control {kind!r}; have {sorted(CONTROLS)}")
    return CONTROLS[kind](conn, seed)

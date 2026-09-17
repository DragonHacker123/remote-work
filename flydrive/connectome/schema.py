"""Signed sparse connectome representation.

The connectome supplies *structure*: who connects to whom, how many synapses,
and the sign of each connection (from the presynaptic neuron's predicted
neurotransmitter). It does not supply synaptic strengths, time constants, or
neuromodulatory state -- those are the free parameters that get learned.

Sign convention follows the standard Drosophila assignment used by whole-brain
LIF models: acetylcholine is excitatory, GABA is inhibitory, and **glutamate is
inhibitory** in the fly (GluCl channels). That last one catches people out.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix

# Neurotransmitter -> sign. Glutamate is inhibitory in Drosophila.
NT_SIGN: dict[str, int] = {
    "acetylcholine": +1,
    "ach": +1,
    "gaba": -1,
    "glutamate": -1,
    "glut": -1,
    # Modulators carry no fast sign of their own; they act through the
    # plasticity rule instead, so they are excluded from the linear operator.
    "dopamine": 0,
    "da": 0,
    "serotonin": 0,
    "5ht": 0,
    "octopamine": 0,
    "oa": 0,
    "unknown": 0,
}


@dataclass
class Connectome:
    """A signed, weighted, directed graph over neurons.

    Attributes
    ----------
    neuron_ids
        Stable external identifiers (FlyWire root ids, or synthetic ids).
    type_ids
        Index into ``type_names``. Cell type is the unit of parameter sharing:
        learned gains, time constants and biases are *per type*, not per
        neuron, mirroring the fact that a genome encodes cell-type-level rules
        rather than 50 million individual synaptic weights.
    region_ids
        Index into ``region_names``. Coarse neuropil / super-class labels.
    side
        -1 left, +1 right, 0 midline.
    pre, post, weight, sign
        Edge list. ``weight`` is raw synapse count, ``sign`` is in {-1, 0, +1}.
    ports
        Named index groups used to inject input and read output, e.g.
        ``ports["EPG"]`` is the heading-ring population.
    """

    neuron_ids: np.ndarray
    type_ids: np.ndarray
    type_names: list[str]
    region_ids: np.ndarray
    region_names: list[str]
    side: np.ndarray
    pre: np.ndarray
    post: np.ndarray
    weight: np.ndarray
    sign: np.ndarray
    ports: dict[str, np.ndarray] = field(default_factory=dict)
    name: str = "connectome"

    # ---------------------------------------------------------------- basics

    @property
    def n_neurons(self) -> int:
        return int(self.neuron_ids.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.pre.shape[0])

    @property
    def n_types(self) -> int:
        return len(self.type_names)

    def type_index(self, name: str) -> int:
        return self.type_names.index(name)

    def neurons_of_type(self, name: str) -> np.ndarray:
        return np.flatnonzero(self.type_ids == self.type_index(name))

    def port(self, name: str) -> np.ndarray:
        if name not in self.ports:
            raise KeyError(f"no port {name!r}; have {sorted(self.ports)}")
        return self.ports[name]

    def summary(self) -> str:
        excit = int((self.sign > 0).sum())
        inhib = int((self.sign < 0).sum())
        mod = int((self.sign == 0).sum())
        return (
            f"{self.name}: {self.n_neurons:,} neurons, {self.n_edges:,} edges, "
            f"{self.n_types} types "
            f"(+{excit:,} / -{inhib:,} / mod {mod:,})"
        )

    def subgraph(self, regions: tuple[str, ...] | None = None, keep: np.ndarray | None = None):
        """Restrict to a set of neuropils (or an explicit neuron mask).

        Used to run the sensorimotor loop without the mushroom body during the
        outer training loop -- the MB is thousands of Kenyon cells that do
        nothing until the plasticity inner loop is switched on, and carrying
        them through every rollout is most of the compute for none of the
        behaviour.
        """
        if keep is None:
            if regions is None:
                raise ValueError("pass either regions or keep")
            wanted = {self.region_names.index(r) for r in regions if r in self.region_names}
            keep = np.isin(self.region_ids, list(wanted))
        keep = np.asarray(keep)
        if keep.dtype != bool:
            mask = np.zeros(self.n_neurons, dtype=bool)
            mask[keep] = True
            keep = mask

        idx = np.flatnonzero(keep)
        remap = np.full(self.n_neurons, -1, dtype=np.int64)
        remap[idx] = np.arange(len(idx))
        edge_ok = (remap[self.pre] >= 0) & (remap[self.post] >= 0)

        ports = {}
        for name, grp in self.ports.items():
            kept = remap[grp]
            kept = kept[kept >= 0]
            if kept.size:
                ports[name] = kept.astype(np.int32)

        return Connectome(
            neuron_ids=self.neuron_ids[idx],
            type_ids=self.type_ids[idx],
            type_names=list(self.type_names),   # keep indices stable
            region_ids=self.region_ids[idx],
            region_names=list(self.region_names),
            side=self.side[idx],
            pre=remap[self.pre[edge_ok]].astype(np.int32),
            post=remap[self.post[edge_ok]].astype(np.int32),
            weight=self.weight[edge_ok],
            sign=self.sign[edge_ok],
            ports=ports,
            name=f"{self.name}|subgraph",
        )

    # ------------------------------------------------------------- operator

    def build_operator(
        self,
        drop_modulatory: bool = True,
        exclude_pairs: list[tuple[str, str]] | None = None,
    ) -> "ConnectomeOperator":
        """Compile the edge list into a fast, rescalable sparse operator.

        ``exclude_pairs`` drops edges between the named (presynaptic type,
        postsynaptic type). Use it to hand a block to another module -- the
        KC->MBON block is excluded here because ``learn.mb`` owns those
        weights and updates them online under dopamine control.
        """
        keep = np.ones(self.n_edges, dtype=bool)
        if drop_modulatory:
            keep = self.sign != 0
        for pre_name, post_name in exclude_pairs or []:
            pt, qt = self.type_index(pre_name), self.type_index(post_name)
            keep &= ~((self.type_ids[self.pre] == pt) & (self.type_ids[self.post] == qt))
        pre = self.pre[keep]
        post = self.post[keep]
        w = self.weight[keep].astype(np.float64) * self.sign[keep]

        n = self.n_neurons
        # Row = postsynaptic, column = presynaptic, so that x_post = W @ r_pre.
        mat = csr_matrix((w, (post, pre)), shape=(n, n))
        mat.sum_duplicates()

        # Row-normalise by total absolute input. This makes a neuron's summed
        # drive independent of its in-degree, which is what keeps a network
        # with a 4-order-of-magnitude degree spread numerically stable.
        absrow = np.abs(mat).sum(axis=1)
        denom = np.asarray(absrow).ravel()
        denom[denom == 0.0] = 1.0
        scale = np.repeat(1.0 / denom, np.diff(mat.indptr))
        base = mat.data * scale

        # Presynaptic cell type of each edge, in CSR order, so gains can be
        # applied with one vectorised multiply.
        edge_pre_type = self.type_ids[mat.indices].astype(np.int32)
        return ConnectomeOperator(
            matrix=mat,
            base_data=base,
            edge_pre_type=edge_pre_type,
            n_types=self.n_types,
        )


@dataclass
class ConnectomeOperator:
    """Sparse operator whose entries rescale by *presynaptic cell type*.

    ``W_ij = sign_j * gain[type(j)] * synapses_ij / total_input_i``
    """

    matrix: csr_matrix
    base_data: np.ndarray
    edge_pre_type: np.ndarray
    n_types: int

    def set_gains(self, gains: np.ndarray) -> csr_matrix:
        """Apply per-type output gains in place and return the matrix."""
        if gains.shape != (self.n_types,):
            raise ValueError(f"expected gains of shape ({self.n_types},), got {gains.shape}")
        np.multiply(self.base_data, gains[self.edge_pre_type], out=self.matrix.data)
        return self.matrix

    def copy(self) -> "ConnectomeOperator":
        return ConnectomeOperator(
            matrix=self.matrix.copy(),
            base_data=self.base_data.copy(),
            edge_pre_type=self.edge_pre_type,
            n_types=self.n_types,
        )

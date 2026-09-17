import numpy as np
import pytest

from flydrive.connectome import apply_control, build_surrogate
from flydrive.connectome.controls import CONTROLS
from flydrive.connectome.schema import NT_SIGN


@pytest.fixture(scope="module")
def conn():
    return build_surrogate(scale=0.5, seed=0)


def test_glutamate_is_inhibitory():
    """The classic Drosophila gotcha: Glu is inhibitory, not excitatory."""
    assert NT_SIGN["glutamate"] == -1
    assert NT_SIGN["gaba"] == -1
    assert NT_SIGN["acetylcholine"] == +1
    assert NT_SIGN["dopamine"] == 0


def test_surrogate_has_required_ports(conn):
    for port in ("heading", "goal", "PFL3L", "PFL3R", "turn_L", "turn_R", "speed", "stop"):
        assert port in conn.ports and len(conn.port(port)) > 0


def test_operator_rows_are_normalised(conn):
    op = conn.build_operator()
    matrix = op.set_gains(np.ones(conn.n_types))
    rows = np.abs(matrix).sum(axis=1).A.ravel()
    nonzero = rows[rows > 0]
    assert np.allclose(nonzero, 1.0), "each neuron's total |input| should be 1"


def test_gains_scale_by_presynaptic_type(conn):
    op = conn.build_operator()
    base = op.set_gains(np.ones(conn.n_types)).copy()
    doubled = op.set_gains(np.full(conn.n_types, 2.0))
    assert np.allclose(doubled.data, 2.0 * base.data)


def test_excluding_a_block_removes_those_edges():
    full = build_surrogate(scale=0.5, seed=0)
    with_kc = full.build_operator()
    without = full.build_operator(exclude_pairs=[("KC", "MBON")])
    assert without.matrix.nnz < with_kc.matrix.nnz


@pytest.mark.parametrize("kind", [k for k in CONTROLS if k != "none"])
def test_controls_preserve_degree_sequences(conn, kind):
    """Every null model must match the real connectome's degrees exactly."""
    shuffled = apply_control(conn, kind, seed=3)
    for attr in ("pre", "post"):
        original = np.bincount(getattr(conn, attr), minlength=conn.n_neurons)
        control = np.bincount(getattr(shuffled, attr), minlength=conn.n_neurons)
        assert np.array_equal(original, control), f"{kind} changed {attr} degrees"


def test_within_type_control_preserves_type_level_connectivity(conn):
    """The strong control: the cell-type-by-cell-type matrix is untouched."""
    shuffled = apply_control(conn, "within-type", seed=5)

    def type_matrix(c):
        key = c.type_ids[c.pre].astype(np.int64) * c.n_types + c.type_ids[c.post]
        return np.bincount(key, minlength=c.n_types**2)

    assert np.array_equal(type_matrix(conn), type_matrix(shuffled))


def test_pairing_control_actually_rewires(conn):
    shuffled = apply_control(conn, "pairing", seed=5)
    assert not np.array_equal(conn.post, shuffled.post)


def test_subgraph_drops_region_and_reindexes(conn):
    regions = tuple(r for r in conn.region_names if r != "MB")
    sub = conn.subgraph(regions=regions)
    assert sub.n_neurons < conn.n_neurons
    assert sub.pre.max() < sub.n_neurons and sub.post.max() < sub.n_neurons
    assert "MB" not in {sub.region_names[i] for i in sub.region_ids}
    # Ports that survive must still point at neurons of the right type.
    assert len(sub.port("heading")) == len(conn.port("heading"))

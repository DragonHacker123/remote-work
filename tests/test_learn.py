import numpy as np
import pytest

from flydrive.connectome import build_surrogate
from flydrive.learn.es import ESConfig, OpenAIES, centered_ranks
from flydrive.learn.mb import MBConfig, MushroomBody


# ------------------------------------------------------------------------ ES


def test_centered_ranks_are_scale_invariant_and_centered():
    for fitness in (np.array([1.0, 2.0, 3.0, 4.0]), np.array([10.0, 1e6, 20.0, 30.0])):
        ranks = centered_ranks(fitness)
        assert ranks.min() == pytest.approx(-0.5)
        assert ranks.max() == pytest.approx(+0.5)
        assert ranks.mean() == pytest.approx(0.0)
        # Order preserved: the best input gets the best rank.
        assert np.argmax(ranks) == np.argmax(fitness)


def test_perturbations_are_antithetic():
    es = OpenAIES(np.zeros(8), ESConfig(popsize=10, seed=1))
    candidates = es.ask()
    half = len(candidates) // 2
    assert np.allclose(candidates[:half] + candidates[half:], 0.0)


def test_es_maximises_a_quadratic():
    target = np.array([0.7, -1.3, 2.0, 0.1])
    es = OpenAIES(np.zeros(4), ESConfig(popsize=24, sigma=0.15, lr=0.15, weight_decay=0.0, seed=0))
    start = np.linalg.norm(es.theta - target)
    for _ in range(220):
        candidates = es.ask()
        es.tell(-np.sum((candidates - target) ** 2, axis=1))
    assert np.linalg.norm(es.theta - target) < 0.2 * start


def test_tell_rejects_a_wrong_sized_population():
    es = OpenAIES(np.zeros(3), ESConfig(popsize=8))
    es.ask()
    with pytest.raises(ValueError):
        es.tell(np.zeros(5))


# ------------------------------------------------------------- mushroom body


@pytest.fixture(scope="module")
def mb():
    conn = build_surrogate(scale=0.5, seed=0)
    return MushroomBody(conn, MBConfig(n_sectors=24, sparsity=10, seed=0))


def test_kenyon_code_is_sparse(mb):
    """APL's gain control is what makes the code sparse. Without sparseness
    every corner's memory overwrites every other corner's."""
    code = mb.kc_code(np.array([0.0, 5.0, 11.5]))
    counts = code.sum(axis=0)
    assert np.all(counts >= mb.cfg.sparsity)
    assert np.all(counts <= mb.cfg.sparsity * 3)   # ties can add a few
    assert counts.max() < 0.05 * mb.n_kc


def test_distant_contexts_get_separable_codes(mb):
    near = mb.kc_code(np.array([3.0]))[:, 0]
    same = mb.kc_code(np.array([3.0]))[:, 0]
    far = mb.kc_code(np.array([15.0]))[:, 0]
    assert np.array_equal(near, same), "coding must be deterministic"
    overlap = float(near @ far) / float(near @ near)
    assert overlap < 0.5, f"codes for distant sectors overlap {overlap:.2f}"


def test_reward_moves_the_readout_toward_the_rewarded_perturbation(mb):
    mb.reset_memory()
    code = mb.kc_code(np.array([7.0]))
    perturb = np.array([[0.4, -0.2]])
    before = mb.output(code)[0]
    mb.update(code, perturb, np.array([1.0]))
    after = mb.output(code)[0]
    assert after[0] > before[0], "rewarded steering trim should be reinforced"
    assert after[1] < before[1]

    # And punishment moves it the other way.
    mb.update(code, perturb, np.array([-2.0]))
    assert mb.output(code)[0][0] < after[0]


def test_learning_stays_local_to_the_context(mb):
    """A memory formed at one corner must not leak to a distant one."""
    mb.reset_memory()
    here = mb.kc_code(np.array([2.0]))
    there = mb.kc_code(np.array([14.0]))
    mb.update(here, np.array([[0.5, 0.0]]), np.array([1.0]))
    assert abs(mb.output(here)[0][0]) > 4 * abs(mb.output(there)[0][0])


def test_dopamine_is_a_prediction_error_against_the_cars_own_best(mb):
    mb.reset_memory()
    sector = np.array([3])
    # First visit has no baseline, so no signal.
    assert mb.dopamine(sector, np.array([2.0]))[0] == pytest.approx(0.0)
    # Slower than before is punished, faster is rewarded.
    assert mb.dopamine(sector, np.array([2.5]))[0] < 0
    assert mb.dopamine(sector, np.array([1.5]))[0] > 0
    assert mb.best_sector_time[3] == pytest.approx(1.5)

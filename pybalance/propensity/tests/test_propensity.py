import numpy as np
from pybalance.propensity import matcher


def test_greedy_without_replacement():
    rw_ptt = np.array(
        [
            0.1,
            0.2,
            0.3,
            0.4,
        ]
    )
    rct_ptt = np.array([0.5, 0.6, 0.7])
    rw, rct = matcher.propensity_score_match(rw_ptt, rct_ptt, method="greedy")
    assert set(rw) == set([1, 2, 3])


def test_lsa():
    rw_ptt = np.array([0.1, 0.2, 0.3, 0.4])
    rct_ptt = np.array([0.5, 0.6, 0.7])
    rw, rct = matcher.propensity_score_match(
        rw_ptt,
        rct_ptt,
        method="linear_sum_assignment",
    )
    assert set(rw) == set([1, 2, 3])


def test_ps_greedy_w_calliper():
    rw_ptt = np.array([0.1, 0.2, 0.3, 0.0, 0.5])
    rct_ptt = np.array([0.2, 0.2, 0.8])
    rw, rct = matcher.propensity_score_match_greedy(rw_ptt, rct_ptt)
    assert all(rw == np.array([1, 2, 4]))
    assert all(rct == np.array([0, 1, 2]))

    rw, rct = matcher.propensity_score_match_greedy(rw_ptt, rct_ptt, caliper=0.1)
    assert all(rw == np.array([1, 2]))
    assert all(rct == np.array([0, 1]))


def test_seed_makes_hyperparam_search_reproducible():
    from pybalance.sim import generate_toy_dataset

    md = generate_toy_dataset()

    def search(seed):
        m = matcher.PropensityScoreMatcher(md, max_iter=6, seed=seed, verbose=False)
        return [(c.__name__, p) for c, p in m._get_hyperparams(6)]

    assert search(0) == search(0)
    assert search(0) != search(1)
    assert matcher.PropensityScoreMatcher(md, seed=3).get_params()["seed"] == 3

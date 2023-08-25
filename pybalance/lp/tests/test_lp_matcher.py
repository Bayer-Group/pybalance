import numpy as np


from pybalance.lp import ConstraintSatisfactionMatcher
from pybalance.sim import generate_toy_dataset


def sum_feature(pool, patients, f):
    patients = [pool[p] for p in patients]
    return sum(p[f] for p in patients)


def test_limits():
    # basically test that the limits chosen by the solver are feasible
    # in the ideal case, ALL subsets should satisfy the constraint.
    matching_data = generate_toy_dataset()

    solver = ConstraintSatisfactionMatcher(matching_data, objective="beta")

    min_features = []
    max_features = []
    max_deltas = []
    match_size = int(len(solver.target) / 2)
    n_features = len(solver.pool[0])

    for f in range(n_features):
        min_feature, max_feature, max_delta = solver._get_limits(
            f, match_size=match_size
        )
        min_features.append(min_feature)
        max_features.append(max_feature)
        max_deltas.append(max_delta)

    patient_pool_list = list(range(len(solver.pool)))
    patient_target_list = list(range(len(solver.target)))
    np.random.seed(1234)
    for _ in range(1000):
        patients_pool = np.random.choice(
            patient_pool_list, replace=False, size=match_size
        )
        patients_target = np.random.choice(
            patient_target_list, replace=False, size=match_size
        )
        for f in range(n_features):
            pool_feature = sum_feature(solver.pool, patients_pool, f)
            target_feature = sum_feature(solver.target, patients_target, f)
            assert min_features[f] <= pool_feature <= max_features[f]
            assert min_features[f] <= target_feature <= max_features[f]
            abs_delta = abs(pool_feature - target_feature)
            assert abs_delta <= max_deltas[f]

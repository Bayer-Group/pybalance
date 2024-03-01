# generate simulated data
import os
import numpy as np
from scipy.stats import truncnorm, uniform
import pandas as pd
from pybalance.utils import MatchingData


def generate_truncated_distributions(rng, mn, mx, size):
    """
    Generate multivariate normal variables with truncations.
        rng = a function which takes as only parameter size and returns
            random variations of shape size x N
        cov = covariance matrix, shape N x N
        mn = lower limits, shape N x 1
        mx = upper limits, shape N x 1
        size = number of samples to draw
    This is implemented by rejection sampling, so if the code runs
    for a really long time, you have probably set parameters for which
    most samples get rejected.
    """
    out = np.empty((0, len(mn)))
    while out.shape[0] < size:
        _out = rng(2 * size)
        where = ((_out <= mx) & (_out >= mn)).min(axis=1)
        if len(where):
            _out = _out[where]
            out = np.vstack([out, _out])

    return out[:size, :]


def multivariate_truncnorm(mu, cov, mn, mx, size):
    """
    Generate multivariate normal variables with truncations.
        mu = means of the covariates, shape N x 1
        cov = covariance matrix, shape N x N
        mn = lower limits, shape N x 1
        mx = upper limits, shape N x 1
        size = number of samples to draw
    This is implemented by rejection sampling, so if the code runs
    for a really long time, you have probably set parameters for which
    most samples get rejected.
    """

    def rng(x):
        return np.random.multivariate_normal(mu, cov, size=x)

    return generate_truncated_distributions(rng, mn, mx, size)


def _truncnorm(mn, mx, mu, std, size):
    """
    For reasons I don't understand, the loc and scale parameters in truncnorm
    don't behave as I expect. Here I reimplment the interface to truncnrom to
    make more sense.
    """
    a, b = (mn - mu) / std, (mx - mu) / std
    return mu + std * truncnorm.rvs(a, b, size=size)


def correlate_vars(x, y, cor):
    """
    Given two uncorrelated covariates x and y return a new random variable with the same
    mean and standard deviation as y but with correlation cor to covariate x.
    """
    mu_0 = y.mean()
    sigma_0 = y.std()
    y = cor * x / x.std() + np.sqrt(1 - cor**2) * y / y.std()
    y = (sigma_0 / y.std()) * (y - y.mean()) + mu_0
    return y


def generate_random_feature_data_rct(
    size, mu_h=150, std_h=20, cor_hw=-0.8, cor_ag=0.8, p_binary=[], seed=42
):
    np.random.seed(seed)

    def rng(size):
        weight = uniform.rvs(loc=50, scale=120 - 50, size=size)
        height = np.random.normal(loc=mu_h, scale=std_h, size=size)
        height = correlate_vars(weight, height, cor_hw)
        return np.vstack([weight, height]).T

    wh = generate_truncated_distributions(rng, mn=[50, 125], mx=[120, 195], size=size)
    weight = wh[:, 0].T
    height = wh[:, 1].T

    def rng(size):
        gender = np.random.choice([0, 1], size=size, replace=True, p=[0.5, 0.5])
        age = np.random.normal(loc=50, scale=20, size=size)
        age = correlate_vars(gender, age, cor_ag)
        return np.vstack([gender, age]).T

    ga = generate_truncated_distributions(rng, mn=[0, 18], mx=[1, 75], size=size)
    gender = ga[:, 0].T
    age = ga[:, 1].T

    haircolor = np.random.choice([0, 1, 2], size=size, replace=True, p=[0.2, 0.4, 0.4])
    country = np.random.choice(
        [0, 1, 2, 3, 4, 5], size=size, replace=True, p=[0.1, 0.2, 0.2, 0.1, 0.2, 0.2]
    )

    data = pd.DataFrame.from_dict(
        {
            "age": age,
            "height": height,
            "weight": weight,
            "gender": gender,
            "haircolor": haircolor,
            "country": country,
            "population": ["target"] * size,
        }
    )
    for i, p in enumerate(p_binary):
        data.loc[:, f"binary_{i}"] = np.array(np.random.rand(size) < p).astype(int)

    return data


def generate_random_feature_data_rwd(
    size, mu_w=90, std_w=20, cor_hw=0.8, cor_ag=-0.8, p_binary=[], seed=43
):
    np.random.seed(seed)

    def rng(size):
        weight = np.random.normal(loc=mu_w, scale=std_w, size=size)
        height = uniform.rvs(loc=125, scale=195 - 125, size=size)
        weight = correlate_vars(height, weight, cor_hw)
        return np.vstack([weight, height]).T

    wh = generate_truncated_distributions(rng, mn=[50, 125], mx=[120, 195], size=size)
    weight = wh[:, 0].T
    height = wh[:, 1].T

    def rng(size):
        gender = np.random.choice([0, 1], size=size, replace=True, p=[0.6, 0.4])
        age = np.random.normal(loc=65, scale=20, size=size)
        age = correlate_vars(gender, age, cor_ag)
        return np.vstack([gender, age]).T

    ga = generate_truncated_distributions(rng, mn=[0, 18], mx=[1, 75], size=size)
    gender = ga[:, 0].T
    age = ga[:, 1].T

    haircolor = np.random.choice([0, 1, 2], size=size, replace=True, p=[0.4, 0.3, 0.3])
    country = np.random.choice(
        [0, 1, 2, 3, 4, 5], size=size, replace=True, p=[0, 0.1, 0.2, 0.3, 0.3, 0.1]
    )

    data = pd.DataFrame.from_dict(
        {
            "age": age,
            "height": height,
            "weight": weight,
            "gender": gender,
            "haircolor": haircolor,
            "country": country,
            "population": ["pool"] * size,
        }
    )

    for i, p in enumerate(p_binary):
        data.loc[:, f"binary_{i}"] = np.array(np.random.rand(size) < p).astype(int)

    return data


def generate_toy_dataset(n_pool=10000, n_target=1000, seed=45):
    """
    Generate a toy matching dataset with n_pool patients in the pool and
    n_target patients in the target population. For finer control, see
    generate_random_feature_data_rwd and generate_random_feature_data_rct.
    """
    pool = generate_random_feature_data_rwd(
        n_pool, cor_hw=0.5, cor_ag=-0.5, p_binary=[0.1, 0.3, 0.5, 0.8], seed=seed
    )
    target = generate_random_feature_data_rct(
        n_target, std_h=20, cor_ag=0.5, p_binary=[0.3, 0.5, 0.3, 0.5], seed=seed + 1
    )
    feature_data = pd.concat([pool, target])
    feature_data.loc[:, "patient_id"] = list(range(len(feature_data)))
    return MatchingData(feature_data)


def get_paper_dataset_path():
    """
    Get the path to the simulated matching dataset presented in the pybalance paper
    (https://onlinelibrary.wiley.com/doi/10.1002/pst.2352).
    """
    filepath = "pool250000-target25000-normal0-lognormal0-binary4.parquet"
    resource = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", filepath
    )
    return resource


def load_paper_dataset():
    """
    Load the simulated matching dataset presented in the pybalance paper
    (https://onlinelibrary.wiley.com/doi/10.1002/pst.2352).
    """
    resource = get_paper_dataset_path()
    return MatchingData(resource)

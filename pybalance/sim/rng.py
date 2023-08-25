# generate simulated data
import numpy as np
from scipy.stats import truncnorm, uniform
import pandas as pd


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
    rng = lambda x: np.random.multivariate_normal(mu, cov, size=x)
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


def generate_random_six_feature_data_rct(
    size, mu_h=150, std_h=20, cor_hw=-0.8, cor_ag=0.8, seed=42
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

    return data


def generate_random_six_feature_data_rwd(
    size, mu_w=90, std_w=20, cor_hw=0.8, cor_ag=-0.8, seed=43
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

    return data

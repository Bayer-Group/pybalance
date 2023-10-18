from typing import Dict, Union, Optional
from collections import defaultdict

import pandas as pd
import numpy as np
import torch

from pybalance.utils import (
    MatchingData,
    split_target_pool,
    BaseMatchingPreprocessor,
    DecisionTreeEncoder,
    StandardMatchingPreprocessor,
    BetaXPreprocessor,
    GammaPreprocessor,
    GammaXPreprocessor,
)

import logging

logger = logging.getLogger(__name__)


def map_input_output_weights(
    preprocessor: BaseMatchingPreprocessor,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Map weights on input features to weights at the output of a given
    preprocessor. This mapping is only possible if the preprocessor defines
    get_feature_names_out(). Weights from the input variable are passed along to
    all output variables. For instance, if "age" is an input variable and gets a
    weight of 10, then each age bin feature will have a weight of 10.

    We also considered "diluting" weights such that the total initial weight is
    spread across the bins. We found, however, that this gives unsatisfactory
    results, since requirements for matching success are usually stated in terms
    of unweighted bins (e.g. |SMD| < 0.1). If for instance age is an important
    feature, then so should any feature constructed from age.
    """
    if weights is None:
        weights = {}
    input_weights = defaultdict(lambda: 1)
    input_weights.update(weights)
    output_weights = defaultdict(lambda: 1)

    for input_col in preprocessor.input_headers["all"]:
        output_cols = preprocessor.get_feature_names_out(input_col)
        weight_in = input_weights[input_col]

        # Pass weight along to derived features
        weight_out = weight_in
        for output_col in output_cols:
            # if we've seen this column before, something is wrong;
            # an output column should correspond to exactly one input column
            assert output_col not in output_weights.keys()
            output_weights[output_col] = weight_out

    assert set(output_weights.keys()) == set(preprocessor.output_headers["all"])

    return output_weights


def reshape_output(f):
    def _f(*args, **kwargs):
        output = f(*args, **kwargs)
        if len(output) == 1:
            return output[0]
        return output

    return _f


class BaseBalanceCalculator:
    """
    BaseBalanceCalculator is the low-level interface to calculating balance.
    BaseBalanceCalculator can be used with any preprocessor defined as a
    subclass of BaseMatchingPreprocessor. BaseBalanceCalculator implements
    matrix calculations in pytorch to allow for GPU acceleration.

    BaseBalanceCalculator performs two main tasks:

      (1) Computes a per-feature-loss based on the output features of the given
          preprocessor and

      (2) Aggregates the per-feature-loss into a single value for the loss.

    Furthermore, the calculator can compute the loss for many populations at a
    time.

    :matching_data: Input matching data to be used for distance calculations.
        Must contain exactly two populations. The smaller population is used as
        a reference population. Calls to distance() compute the distance to this
        reference population.

    :preprocessor: Preprocessor to use for per-feature-loss calculation. The
        per-feature-loss is, up to some normalizations, the mean difference in
        the features at the output of the preprocessor.

    :feature_weights: How to weight features in aggregation of per-feature-loss.

    :order: Exponent to use in combining per-feature-loss into an aggregate
        loss. Total loss is sum(feature_weight * feature_loss**order)**(1/order).

    :param standardize_difference: Whether to use the absolute standardized mean
        difference for the per-feature loss (otherwise uses absolute mean
        difference).

    :device: Name of device to use for matrix computations. By default, will use
        GPU if a GPU is found on the system.
    """

    name = "base"

    def __init__(
        self,
        matching_data: MatchingData,
        preprocessor: BaseMatchingPreprocessor,
        feature_weights: Optional[Dict[str, float]] = None,
        order: float = 1,
        standardize_difference: bool = True,
        device: Optional[str] = None,
    ):
        self.order = order
        self.standardize_difference = standardize_difference
        self.device = self._get_device(device)
        self.preprocessor = preprocessor
        self.preprocessor.fit(matching_data)
        self.matching_data = matching_data
        target, pool = split_target_pool(matching_data)
        self.target = self._preprocess(target)
        self.pool = self._preprocess(pool)
        self._set_feature_weights(feature_weights)

        self.target_mean = torch.mean(self.target, 0, True).to(self.device)
        self.target_std = torch.std(self.target, 0, keepdim=True).to(self.device)

        # Zero variances are bad and can lead to infinite loss.
        if any((self.target_std == 0)[0]):
            bad_columns = [
                self.preprocessor.output_headers["all"][j]
                for j, std in enumerate(self.target_std[0])
                if std == 0
            ]
            logger.warning(
                f'Detected constant feature(s) in target population: {",".join(bad_columns)}.'
            )

        pool_std = torch.std(self.pool, 0, keepdim=True).to(self.device)
        # Zero variances are bad and can lead to infinite loss.
        if any((pool_std == 0)[0]):
            bad_columns = [
                self.preprocessor.output_headers["all"][j]
                for j, std in enumerate(pool_std[0])
                if std == 0
            ]
            logger.warning(
                f'Detected constant feature(s) in pool population: {",".join(bad_columns)}.'
            )

    def _set_feature_weights(self, feature_weights):
        if feature_weights is None:
            feature_weights = {}
            try:
                feature_weights = map_input_output_weights(
                    self.preprocessor, feature_weights
                )
            except NotImplementedError:
                # User has not passed weights, so we can assume that
                # NotImplementedError can be ignored. Note that not passing
                # weights does not mean weights will always be equal!! Equal
                # weights on the input get diluted on the output, depending on
                # how many output features an input feature is mapped to. In the
                # current case, there simply is no mapping available, so we must
                # put equal weights on the output feature space.
                feature_weights = defaultdict(lambda: 1)
        else:
            # User has explicitly passed weights, so NotImplementedError should
            # not be handled
            feature_weights = map_input_output_weights(
                self.preprocessor, feature_weights
            )

        feature_weights = np.array(
            [feature_weights[c] for c in self.preprocessor.output_headers["all"]]
        )
        feature_weights = feature_weights / sum(feature_weights)
        self.feature_weights = torch.tensor(feature_weights).to(self.device)

    def _get_device(self, device):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            else:
                device = "cpu"
        return torch.device(device)

    def _preprocess(self, data: Union[pd.DataFrame, MatchingData]):
        if isinstance(data, pd.DataFrame):
            data = MatchingData(
                data=data,
                headers=self.matching_data.headers,
                population_col=self.matching_data.population_col,
            )
        data = self.preprocessor.transform(data)
        data = data[data.headers["all"]].values
        # It's best to store the feature data on the GPU to avoid moving it back
        # and forth. Even for large populations, this will be cheap (e.g.
        # 500,000 patients with 100 features = less than 1GB). Be careful,
        # however: when slicing for the candidate populations, this can blow up
        # the memory footprint. Use the BatchedBalanceCalculator to stay within
        # memory limits.
        data = data.astype(np.float32)
        data = torch.tensor(
            data, dtype=torch.float32, device=self.device, requires_grad=False
        )
        return data

    def _fetch_pool_features(self, candidate_populations):
        if isinstance(candidate_populations, pd.DataFrame):
            features = self._preprocess(candidate_populations)
        else:
            if not isinstance(candidate_populations, torch.Tensor):
                candidate_populations = torch.tensor(
                    candidate_populations, device=self.device, requires_grad=False
                )
            features = self.pool[candidate_populations]

        # features has shape n_candidate_populations x n_patients x n_features
        # note that using the reshape() method works for both numpy and pytorch
        if len(features.shape) == 2:
            features = features.reshape(1, *features.shape)

        return features

    def _finalize_batches(self, batches):
        """
        Combine a list (batches) of batched distances calculations into a single
        pytorch tensor.
        """
        return torch.hstack(batches)

    def _to_array(self, candidate_populations):
        return torch.tensor(candidate_populations, device=self.device)

    def _to_list(self, candidate_populations):
        return candidate_populations.cpu().detach().numpy().tolist()

    @reshape_output
    def distance(
        self, candidate_populations: Union[pd.DataFrame, torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        """
        Compute overall distance (aka "mismatch", aka "loss") between input
        candidate populations. The per-feature loss is aggregated using a vector
        norm specified by order specified in __init__().

        :candidate_populations: Populations for which to compute the mismatch.
            Input can be either a single pandas dataframe containing all the
            required feature data or a 2-dimensional integer array whose entries
            are the indices of patients in the pool. In the latter case, the
            array should be an array of shape n_candidate_populations x
            candidate_population_size that indexes the patient pool (passed
            during __init__()). The first dimension may be omitted if only one
            candidate population is present.
        """
        per_feature_loss = self.per_feature_loss(candidate_populations)
        # Since feature_weights sum to 1, sum is actually a weighted mean
        return torch.sum(
            self.feature_weights * torch.abs(per_feature_loss) ** self.order, dim=1
        ) ** (1.0 / self.order)

    def balance(self, candidate_populations):
        return -self.distance(candidate_populations)

    def per_feature_loss(
        self, candidate_populations: Union[pd.DataFrame, torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        """
        Compute mismatch (aka "distance", aka "loss") on a per-feature basis for
        a set of candidate populations.
        """
        pool = self._fetch_pool_features(candidate_populations)

        # NaNs can arise when the pool and target both have zero variance. If
        # they have zero variance and are the same value (e.g. pool all 0s and
        # target all 0s), then this should be anyway zero loss. As long as the
        # norm is non-zero, we are good. If they have zero variance and are
        # different values (pool all 1s and target all 0s), then this should
        # represent a large loss. Technically, the loss should be infinite in
        # that case, but infinities are annoying so we just set the norm to a
        # small positive number. The user will get warned in the call to
        # __init__() of the possibility of this arising.
        norm = 1
        if self.standardize_difference:
            norm *= torch.sqrt(pool.std(axis=1) ** 2 + self.target_std**2) + 1e-6

        return torch.nan_to_num((pool.mean(axis=1) - self.target_mean) / norm)


class BetaBalance(BaseBalanceCalculator):
    """
    Convenience interface to BaseBalanceCalculator to computes the distance
    between populations as the mean standardized mean difference. Uses
    StandardMatchingPreprocessor as the preprocessor.
    """

    name = "beta"

    def __init__(
        self,
        matching_data: MatchingData,
        feature_weights: Optional[Dict[str, float]] = None,
        device: Optional[str] = None,
        drop: bool = "first",
        standardize_difference: bool = True,
    ):
        preprocessor = StandardMatchingPreprocessor(drop=drop)
        super(BetaBalance, self).__init__(
            matching_data=matching_data,
            preprocessor=preprocessor,
            feature_weights=feature_weights,
            order=1,
            standardize_difference=standardize_difference,
            device=device,
        )


class BetaSquaredBalance(BaseBalanceCalculator):
    """
    Same as BetaBalance, except that per-feature balances are averaged in a
    mean square fashion.
    """

    name = "beta_squared"

    def __init__(
        self,
        matching_data: MatchingData,
        feature_weights: Optional[Dict[str, float]] = None,
        device: Optional[str] = None,
        drop: bool = "first",
        standardize_difference: bool = True,
    ):
        preprocessor = StandardMatchingPreprocessor(drop=drop)
        super(BetaSquaredBalance, self).__init__(
            matching_data=matching_data,
            preprocessor=preprocessor,
            feature_weights=feature_weights,
            order=2,
            standardize_difference=standardize_difference,
            device=device,
        )


class BetaXBalance(BaseBalanceCalculator):
    """
    Convenience interface to BaseBalanceCalculator to compute the balance
    between two populations by computing the standardized mean difference,
    including cross terms. See BetaXPreprocessor for description of
    preprocessing options.
    """

    name = "beta_x"

    def __init__(
        self,
        matching_data: MatchingData,
        feature_weights: Optional[Dict[str, float]] = None,
        device: Optional[str] = None,
        drop: str = "first",
        standardize_difference: bool = True,
        max_cross_terms="auto",
    ):
        pp_x = BetaXPreprocessor(
            drop=drop,
            max_cross_terms=max_cross_terms,
        )
        super(BetaXBalance, self).__init__(
            matching_data=matching_data,
            preprocessor=pp_x,
            order=1,
            standardize_difference=standardize_difference,
            device=device,
        )


class BetaXSquaredBalance(BaseBalanceCalculator):
    """
    Same as BetaXBalance, except that per-feature balances are averages in a
    mean square fashion.
    """

    name = "beta_x_squared"

    def __init__(
        self,
        matching_data: MatchingData,
        feature_weights: Optional[Dict[str, float]] = None,
        device: Optional[str] = None,
        drop: str = "first",
        standardize_difference: bool = True,
        max_cross_terms="auto",
    ):
        pp_x = BetaXPreprocessor(
            drop=drop,
            max_cross_terms=max_cross_terms,
        )
        super(BetaXSquaredBalance, self).__init__(
            matching_data=matching_data,
            preprocessor=pp_x,
            order=2,
            standardize_difference=standardize_difference,
            device=device,
        )


class BetaMaxBalance(BaseBalanceCalculator):
    """
    Same as BetaBalance, except the worst-matched feature determines the loss.
    This class is provided as a convenience, since this balance metric is often
    a criterion used to determine if matching is "sufficiently good". However,
    be aware that using this balance metric as an optimization objective with
    the various matchers can lead unwanted behavior, since if improvements in
    the worst-matched feature are not possible, there is no signal from the
    balance function to improve any other the other features.
    """

    name = "beta_max"

    def __init__(
        self,
        matching_data: MatchingData,
        feature_weights: Optional[Dict[str, float]] = None,
        device: Optional[str] = None,
        drop: bool = "first",
        standardize_difference: bool = True,
    ):
        preprocessor = StandardMatchingPreprocessor(drop=drop)
        super(BetaMaxBalance, self).__init__(
            matching_data=matching_data,
            preprocessor=preprocessor,
            feature_weights=feature_weights,
            order=1,
            standardize_difference=standardize_difference,
            device=device,
        )

    @reshape_output
    def distance(
        self, candidate_populations: Union[pd.DataFrame, torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        per_feature_loss = self.per_feature_loss(candidate_populations)

        return torch.max(
            self.feature_weights * torch.abs(per_feature_loss), dim=1
        ).values


class GammaBalance(BaseBalanceCalculator):
    """
    Convenience interface to BaseBalanceCalculator to compute the balance
    between two populations by computing the mean area between their
    one-dimensional marginal distributions. See GammaPreprocessor for
    description of preprocessing options.
    """

    name = "gamma"

    def __init__(
        self,
        matching_data: MatchingData,
        feature_weights: Optional[Dict[str, float]] = None,
        device: Optional[str] = None,
        n_bins: int = 5,
        encode: str = "onehot-dense",
        cumulative: bool = True,
        drop: str = "first",
        standardize_difference: bool = True,
    ):
        preprocessor = GammaPreprocessor(
            n_bins=n_bins, encode=encode, cumulative=cumulative, drop=drop
        )
        super(GammaBalance, self).__init__(
            matching_data=matching_data,
            preprocessor=preprocessor,
            feature_weights=feature_weights,
            order=1,
            standardize_difference=standardize_difference,
            device=device,
        )


class GammaSquaredBalance(BaseBalanceCalculator):
    """
    Same as GammaBalance, except that per-feature balances are averages in a
    mean square fashion.
    """

    name = "gamma_squared"

    def __init__(
        self,
        matching_data: MatchingData,
        feature_weights: Optional[Dict[str, float]] = None,
        device: Optional[str] = None,
        n_bins: int = 5,
        encode: str = "onehot-dense",
        cumulative: bool = True,
        drop: str = "first",
        standardize_difference: bool = True,
    ):
        preprocessor = GammaPreprocessor(
            n_bins=n_bins, encode=encode, cumulative=cumulative, drop=drop
        )
        super(GammaSquaredBalance, self).__init__(
            matching_data=matching_data,
            preprocessor=preprocessor,
            feature_weights=feature_weights,
            order=2,
            standardize_difference=standardize_difference,
            device=device,
        )


class GammaXBalance(BaseBalanceCalculator):
    """
    Convenience interface to BaseBalanceCalculator to compute the balance
    between two populations by computing the mean area between their
    one-dimensional marginal distributions, including cross terms. See
    GammaXPreprocessor for description of preprocessing options.
    """

    name = "gamma_x"

    def __init__(
        self,
        matching_data: MatchingData,
        feature_weights: Optional[Dict[str, float]] = None,
        device: Optional[str] = None,
        n_bins: int = 5,
        encode: str = "onehot-dense",
        cumulative: bool = True,
        drop: str = "first",
        standardize_difference: bool = True,
        max_cross_terms="auto",
    ):
        pp_x = GammaXPreprocessor(
            n_bins=n_bins,
            encode=encode,
            cumulative=cumulative,
            drop=drop,
            max_cross_terms=max_cross_terms,
        )
        super(GammaXBalance, self).__init__(
            matching_data=matching_data,
            preprocessor=pp_x,
            order=1,
            standardize_difference=standardize_difference,
            device=device,
        )


class GammaXTreeBalance(BaseBalanceCalculator):
    name = "gamma_x_tree"

    def __init__(
        self,
        matching_data,
        keep_original_features=False,
        device=None,
        standardize_difference: bool = True,
        **decision_tree_params,
    ):
        pp_tree = DecisionTreeEncoder(
            keep_original_features=keep_original_features, **decision_tree_params
        )
        super(GammaXTreeBalance, self).__init__(
            matching_data=matching_data,
            preprocessor=pp_tree,
            order=1,
            standardize_difference=standardize_difference,
            device=device,
        )


def _get_batch_size(target_population_size, n_features, max_batch_size_gb=8):
    """
    Get the size of batches for balance calculations such that no batch is
    greater in memory footprint than max_size_gb GB. This calculation is only
    approximate and not guaranteed to respect size requirements. n_features
    should be interpreted as the number of *effective* features (i.e. after
    binning)
    """
    max_batch_size_gb = max_batch_size_gb / 1.25  # add some wiggle room
    size_of_float = (
        4  # Balance calculators are supposed to transform to float datatypes!
    )
    # Technically, the size is given by:
    # batch_size_gb = n_candidate_populations * target_population_size * n_features * size_of_float / 2**30
    # But this value will cancel out in the next step.
    batch_size_gb = target_population_size * n_features * size_of_float / 2**30
    batch_size = int(max_batch_size_gb / batch_size_gb)

    # If you fail this assertion, something is definitely wrong and you'll pay
    # for it downstream. Rather stop you here. Basically, it means you can't
    # even hold 10 candidate populations in memory at a time.
    assert batch_size >= 10
    return batch_size


class BatchedBalanceCaclulator:
    """
    Batch balance calculations to avoid large peak memory usage.
    """

    def __init__(self, balance_calculator, max_batch_size_gb=8):
        self.name = balance_calculator.name
        self.matching_data = balance_calculator.matching_data
        self.preprocessor = balance_calculator.preprocessor
        self.pool = balance_calculator.pool
        self.target = balance_calculator.target
        self.balance_calculator = balance_calculator
        self.max_batch_size_gb = max_batch_size_gb

    def _to_array(self, candidate_populations):
        return self.balance_calculator._to_array(candidate_populations)

    def _to_list(self, candidate_populations):
        return self.balance_calculator._to_list(candidate_populations)

    def distance(self, candidate_populations):
        # If the user passes a pandas DataFrame, the DataFrame is assumed to
        # represent feature data for only one population and no need for
        # batching; just pass on to the base class. Otherwise, assume
        # candidate_populations refers to patient indices.
        if isinstance(candidate_populations, pd.DataFrame):
            return self.balance_calculator.distance(candidate_populations)

        # If the user passes a list, convert it to the underlying backend array
        # type for further processing.
        if isinstance(candidate_populations, list):
            candidate_populations = self.balance_calculator._to_array(
                candidate_populations
            )

        # If array has only one dimension, no need for batching, just pass along
        # to base class.
        if len(candidate_populations.shape) == 1:
            return self.balance_calculator.distance(candidate_populations)

        # Get batch size according to number of size of target population
        batch_size = _get_batch_size(
            self.balance_calculator.target.shape[0],
            self.balance_calculator.target.shape[1],
            self.max_batch_size_gb,
        )

        # From here on, can assume candidate_populations is a 2D array of
        # patient indices with the 0th dimension corresponding to the
        # candidate_population and the 1st dimension corresponding to patient.
        n_remaining = len(candidate_populations)
        distances = []
        j = 0
        while n_remaining > 0:
            N = min(batch_size, n_remaining)
            cps = candidate_populations[j * batch_size : j * batch_size + N, :]
            distances.append(self.balance_calculator.distance(cps))
            n_remaining -= N
            j += 1

        distances = self.balance_calculator._finalize_batches(distances)
        assert len(distances) == len(candidate_populations)
        return distances

    def balance(self, candidate_populations):
        return -self.distance(candidate_populations)


#
# Convenience interface to balance calculators
#
BALANCE_CALCULATORS = {
    BaseBalanceCalculator.name: BaseBalanceCalculator,
    BetaBalance.name: BetaBalance,
    BetaXBalance.name: BetaXBalance,
    BetaXSquaredBalance.name: BetaXSquaredBalance,
    BetaMaxBalance.name: BetaMaxBalance,
    BetaSquaredBalance.name: BetaSquaredBalance,
    GammaBalance.name: GammaBalance,
    GammaSquaredBalance.name: GammaSquaredBalance,
    GammaXTreeBalance.name: GammaXTreeBalance,
    GammaXBalance.name: GammaXBalance,
}


def BalanceCalculator(matching_data, objective="gamma", **kwargs):
    """
    BalanceCalculator provides a convenience interface to balance calculators,
    allowing the user to initialize a balance calculator by name. The
    calculators are initialized with default parameters, but these can be
    overridden by passing the appropriate kwargs.

    :param matching_data: MatchingData instance containing reference to the data
        against which matching metrics will be computed

    :param objective: Name of objective function to be used for computing
        balance. Balance calculators must be implemented in
        utils.balance_calculators.py and registered in the BALANCE_CALCULATORS
        dictionary therein in order to be accessible from this interface.

    :param kwargs: Any additional arguments required to configure the specific
        objective function (e.g. n_bins = 10 for "gamma").
    """
    if objective not in BALANCE_CALCULATORS.keys():
        raise ValueError(
            f"Unknown objective function {objective}. Must be one of {','.join(BALANCE_CALCULATORS.keys())}"
        )
    balance_calculator = BALANCE_CALCULATORS[objective]
    return balance_calculator(matching_data, **kwargs)

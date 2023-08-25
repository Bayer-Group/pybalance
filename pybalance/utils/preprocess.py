from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional, Tuple

from collections import defaultdict
import pandas as pd
import numpy as np
import re
import copy

from sklearn.preprocessing import (
    OrdinalEncoder,
    KBinsDiscretizer,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.compose import ColumnTransformer
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

from pybalance.utils.matching_data import MatchingData, MatchingHeaders
from pybalance.utils.misc import require_fitted

import logging

logger = logging.getLogger(__name__)


class BaseMatchingPreprocessor(ABC):
    """
    BaseMatchingPreprocessor is an abstract preprocessor class for organizing
    data transformations on matching data, keeping track of all preprocessing
    steps so that data are always transformed transformed in the same way.

    The inherited class must implement:

        _fit(), _transform(), _get_output_headers()

    and should implement, if possible,

        _get_feature_names_out()

    The class extends the preprocessing classes defined in sklearn. In addition
    to handling transformations of the data, this class also handles logic of
    MatchingHeaders. Preprocessors which conform to the BaseMatchingPreprocessor
    standard are chainable, allowing one to easily combine preprocessing tasks;
    see ChainPreprocessor.
    """

    def __init__(self):
        self.is_fitted = False

    def fit(self, matching_data: MatchingData, refit: bool = False) -> None:
        """
        A simple wrapper around the workhorse method _fit() to mark the
        preprocessor as fitted after a call to fit() and prevent fitting twice.
        """
        if self.is_fitted and not refit:
            raise RuntimeError(
                f"Preprocessor {self.__class__.__name__} already fitted!"
            )

        self.input_headers = matching_data.headers
        self._fit(matching_data)
        self.is_fitted = True

    @require_fitted
    def transform(self, matching_data: MatchingData) -> MatchingData:
        return self._transform(matching_data)

    def fit_transform(
        self, matching_data: MatchingData, refit: bool = False
    ) -> MatchingData:
        if not self.is_fitted:
            self.fit(matching_data, refit)
        return self.transform(matching_data)

    @abstractmethod
    def _fit(self, matching_data: MatchingData) -> None:
        """
        This method performs all calculations needed in order to perform the
        transformation tasks of the preprocessor (e.g. computing means and
        standard deviations). Should accept a MatchingData instance as input and
        return None. This method must be overridden by the subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def _transform(self, matching_data: MatchingData) -> MatchingData:
        """
        Transform matching_data. Should accept a MatchingData instance as input
        and return a transformed MatchingData instance, including in particular
        transformed metadata. This method must be overridden by the subclass.
        """
        raise NotImplementedError

    @property
    @require_fitted
    def output_headers(self):
        """
        Returns the matching headers which describe the transformed features.
        """
        return self._get_output_headers()

    @abstractmethod
    def _get_output_headers(self):
        """
        Return headers on the output matching data. This method must be
        overridden by the subclass.
        """
        raise NotImplementedError

    @require_fitted
    def get_feature_names_out(
        self, feature_name_in: Optional[Union[str, List[str]]] = None
    ) -> List[str]:
        """
        Map features on an input dataframe to the corresponding set
        of output feature names. This utility is required in certain cases,
        but most notably by balance calculators in weighting features. The
        balance calculator is provided weights on input features and needs
        to know how to map these weights to features output by the preprocessor.

        This mapping is not always possible. For instance, in the tree encoder,
        constructed features are combinations of multiple input features. When
        this mapping is not possible, neither is feature weighting and such
        preprocessors should just leave this method unimplemented.

        NB: sklearn defines similar methods but they do not help us here
        because their methods only accept the entire set of input feature names;
        here we need to be able to map a _single_ feature.
        """
        if feature_name_in is None:
            feature_name_in = self.input_headers["all"]
        if isinstance(feature_name_in, str):
            feature_name_in = [feature_name_in]
        return sum([self._get_feature_names_out(c) for c in feature_name_in], [])

    def _get_feature_names_out(self, feature_name_in: str) -> List[str]:
        """
        Same as get_feature_names_out but on a single feature level. This method
        should be overridden by the subclass.
        """
        raise NotImplementedError


class FloatEncoder(BaseMatchingPreprocessor):
    """
    FloatEncoder converts all variables to float. In particular, if a certain
    variable is encoded as a string (e.g. gender = 'Male'), FloatEncoder will
    transform that variable to a numeric representation (e.g. gender = 0). It
    does so in such a way that the same numeric value will always represent
    the same input string value on repeated calls to transform().
    """

    def __init__(self):
        self.ordinal_encoder = OrdinalEncoder()
        super(FloatEncoder, self).__init__()

    def _fit(self, matching_data: MatchingData) -> None:
        self.nonfloat_columns = []
        for c in matching_data.headers["categoric"]:
            try:
                matching_data[c].astype(float)
            except ValueError:
                self.nonfloat_columns.append(c)
        self.ordinal_encoder.fit(matching_data[self.nonfloat_columns])

    def _get_output_headers(self):
        # Headers are unaffected by the transformation
        return self.input_headers

    def _transform(self, matching_data: MatchingData) -> MatchingData:
        # Make a copy of the input data, to which we will write the transformed data
        data = matching_data.copy().data

        # Convert nonfloats to numeric
        input_columns = self.nonfloat_columns
        output_columns = self.get_feature_names_out(input_columns)
        data.loc[:, output_columns] = self.ordinal_encoder.transform(
            data[input_columns]
        )

        # Should be now possible to cast all features to float
        input_columns = self.input_headers["all"]
        output_columns = self.get_feature_names_out(input_columns)
        data.loc[:, output_columns] = data[input_columns].astype(float)

        # Transform to MatchingData type
        matching_data = MatchingData(
            data=data,
            headers=self.output_headers,
            population_col=matching_data.population_col,
        )

        return matching_data

    def _get_feature_names_out(self, feature_name_in: str) -> List[str]:
        return [feature_name_in]


class CategoricOneHotEncoder(BaseMatchingPreprocessor):
    """
    CategoricOneHotEncoder converts categoric covariates into one-hot encoded
    variables. Numeric columns are unaffected.

    :param drop: Which, if any, columns to drop in the transformation. Choices
        are: {‘first’, ‘if_binary’, None}. See
        sklearn.preprocessing.OneHotEncoder for more details.
    """

    def __init__(self, drop: Optional[str] = "first"):
        super(CategoricOneHotEncoder, self).__init__()
        self.onehot_encoder = OneHotEncoder(drop=drop, sparse_output=False)

    def _fit(self, matching_data: MatchingData) -> None:
        self.onehot_encoder.fit(matching_data[matching_data.headers["categoric"]])

    def _get_output_headers(self):
        # All categoric variables get mapped to categoric; numeric variables
        # are unaffected
        return MatchingHeaders(
            categoric=self.get_feature_names_out(self.input_headers["categoric"]),
            numeric=self.input_headers["numeric"],
        )

    def _transform(self, matching_data: MatchingData) -> MatchingData:
        # Make a copy of the input data, to which we will write the transformed data
        data = matching_data.copy().data

        # Transform categoric variables
        input_columns = self.input_headers["categoric"]
        output_columns = self.get_feature_names_out(input_columns)
        data.loc[:, output_columns] = self.onehot_encoder.transform(data[input_columns])

        # Transform to MatchingData type
        matching_data = MatchingData(
            data=data,
            headers=self.output_headers,
            population_col=matching_data.population_col,
        )
        return matching_data

    def _get_feature_names_out(self, feature_name_in: str) -> List[str]:
        if feature_name_in in self.input_headers["numeric"]:
            return [feature_name_in]

        feature_index_in = self.onehot_encoder.feature_names_in_.tolist().index(
            feature_name_in
        )
        n_categories = sum(
            [len(c) for c in self.onehot_encoder.categories_[:feature_index_in]]
        )
        if self.onehot_encoder.drop is not None:
            dropped_features = sum(
                [
                    1 if d is not None else 0
                    for d in self.onehot_encoder.drop_idx_[:feature_index_in]
                ]
            )
        else:
            dropped_features = 0
        feature_index_out_start = n_categories - dropped_features

        n_categories = len(self.onehot_encoder.categories_[feature_index_in])
        if self.onehot_encoder.drop is not None:
            dropped_features = (
                1 if self.onehot_encoder.drop_idx_[feature_index_in] is not None else 0
            )
        else:
            dropped_features = 0
        n_output_features = n_categories - dropped_features

        feature_index_out_end = feature_index_out_start + n_output_features
        feature_names_out = self.onehot_encoder.get_feature_names_out()[
            feature_index_out_start:feature_index_out_end
        ].tolist()

        assert all(feature_name_in in f for f in feature_names_out)

        return feature_names_out


class NumericBinsEncoder(BaseMatchingPreprocessor):
    """
    NumericBinsEncoder discretizes numeric covariates according to specified
    binning strategy. Categoric columns are unaffected.

    :param n_bins: Number of bins to split numeric variable into. Note in the
        case of cumulative = True, the last bin would be always one. To avoid
        this, internally we use n_bins + 1 bins and drop the last bin.

    :param strategy: Strategy to use for binnings. Choices are: {‘uniform’,
        ‘quantile’, ‘kmeans’}. See sklearn.preprocessing.KBinsDiscretizer for
        more details.

    :param encode: Method to use for encoding numeric variable. Choices are:
        {‘onehot’, ‘onehot-dense’, ‘ordinal’}. See
        sklearn.preprocessing.KBinsDiscretizer for more details.

    :param cumulative: Whether to transform numeric variables to discretized
        cumulative distribution. E.g. if x is in the 3rd bin and n_bins = 4,
        then x will map to [0, 0, 1, 0] if cumulative = False or [0, 0, 1, 1]
        otherwise.
    """

    def __init__(
        self,
        n_bins: int = 5,
        strategy: str = "uniform",
        encode: str = "onehot-dense",
        cumulative: bool = False,
    ) -> None:
        if encode == "onehot":
            logger.warning("Sparse encoding not supported. Using dense instead.")
            encode = "onehot-dense"
        if encode != "onehot-dense" and cumulative:
            raise ValueError(
                "Cumulative = True only valid option for onehot-dense encoding."
            )

        self.cumulative = cumulative
        if self.cumulative:
            # last bin will be all ones and will get dropped, so add an extra bin
            # so user gets what was requested
            n_bins += 1
        self.discretizer = KBinsDiscretizer(
            n_bins=n_bins, strategy=strategy, encode=encode
        )
        super(NumericBinsEncoder, self).__init__()

    def _fit(self, matching_data: MatchingData) -> None:
        if matching_data.headers["numeric"]:
            self.discretizer.fit(matching_data[matching_data.headers["numeric"]])
            for j, c in enumerate(matching_data.headers["numeric"]):
                bin_edges = self.discretizer.bin_edges_
                bin_edges_str = ", ".join(map(str, bin_edges[j].round(2)))
                logger.info(f"Discretized {c} with bins [{bin_edges_str}].")

    def _get_output_headers(self):
        # All output variables are categoric
        return MatchingHeaders(
            categoric=self.get_feature_names_out(),
            numeric=[],
        )

    def _transform(self, matching_data: MatchingData) -> MatchingData:
        # Make a copy of the input data, to which we will add the transformed data
        data = matching_data.copy().data

        # Transform numeric variables
        input_columns = self.input_headers["numeric"]
        output_columns = self.discretizer.get_feature_names_out()
        data.loc[:, output_columns] = self.discretizer.transform(data[input_columns])

        if self.cumulative:
            # Whoops, we have added an extra column! This will
            # happen only when cumulative = True because we train discretizer
            # on self.n_bins + 1 bins since last bin is all ones
            expected_output_columns = self.get_feature_names_out()
            for col in set(output_columns) - set(expected_output_columns):
                del data[col]

            # cumsum data
            for col in input_columns:
                these_output_columns = self.get_feature_names_out(col)
                data.loc[:, these_output_columns] = data[these_output_columns].cumsum(
                    axis=1
                )

        matching_data = MatchingData(
            data=data,
            headers=self.output_headers,
            population_col=matching_data.population_col,
        )
        return matching_data

    def _get_feature_names_out(self, feature_name_in: str) -> List[str]:
        if feature_name_in in self.input_headers["categoric"]:
            return [feature_name_in]

        if self.discretizer.encode == "ordinal":
            return [feature_name_in]

        # We have to make some assumptions here about what sklearn is doing under the hood.
        # Basically, we are assuming that the order of features out from sklearn's
        # get_feature_names_out() is concordant with the order of features in. The assumption is
        # valid as of sklearn 1.2.1 and the assertion below is meant to check that
        # things haven't gone off the rails.
        feature_index_in = self.discretizer.feature_names_in_.tolist().index(
            feature_name_in
        )

        feature_index_out_start = sum(self.discretizer.n_bins_[:feature_index_in])
        n_output_features = self.discretizer.n_bins_[feature_index_in]

        if self.cumulative:
            # last column is all ones, so drop it
            n_output_features -= 1

        feature_index_out_end = feature_index_out_start + n_output_features

        feature_names_out = self.discretizer.get_feature_names_out()[
            feature_index_out_start:feature_index_out_end
        ].tolist()
        assert all(feature_name_in in f for f in feature_names_out)

        return feature_names_out


class CrossTermsPreprocessor(BaseMatchingPreprocessor):
    """
    CrossTermsPreprocessor adds terms of the form:

        (X - <X>) * (Y - <Y>) / ( std(X) * std(Y) )

    as effective features to transformed data. Mean subtraction and scaling by
    the standard deviation is only applied to numeric covariates.

    :param max_cross_terms: Maximum number of cross terms to add. Cross terms
        are ranked by the correlation of the component variables and only the
        max_new_features most correlated pairs are added. If None, all cross
        terms are added. If 'auto', then a heuristic will be applied to pick a
        reasonable number of cross terms.
    """

    def __init__(
        self, subtract_mean: bool = True, max_cross_terms: Union[str, int] = "auto"
    ):
        self.max_cross_terms = max_cross_terms
        if isinstance(self.max_cross_terms, str) and self.max_cross_terms != "auto":
            raise ValueError(
                f"Unrecognized choice for max_cross_terms: {max_cross_terms}"
            )

        self.scaler = StandardScaler()
        super(CrossTermsPreprocessor, self).__init__()

    def _fit(self, matching_data: MatchingData) -> None:
        if self.max_cross_terms == "auto":
            self.max_cross_terms = len(matching_data.headers["all"])
        self.scaler.fit_transform(matching_data[matching_data.headers["numeric"]])
        self.feature_pairs = self._get_most_correlated_pairs(
            matching_data[matching_data.headers["all"]]
        )

    def _get_most_correlated_pairs(self, data: pd.DataFrame) -> Tuple[str]:
        # Caller should have removed all non-feature columns
        headers = data.columns.values.tolist()
        cov = data.corr().abs().values
        cov = cov - np.diag(np.diag(cov))
        # Pandas may drop columns with nans or non-numeric types, which we don't
        # expect to happen, so throw an error if it does.
        assert cov.shape == (len(headers), len(headers))
        indices = [np.unravel_index(k, cov.shape) for k in cov.flatten().argsort()][
            ::-1
        ]
        correlated_pairs = [
            (headers[i], headers[j]) for i, j in indices if i < j
        ]  # remove symmetric pairs

        if self.max_cross_terms is not None:
            correlated_pairs = correlated_pairs[: self.max_cross_terms]

        return correlated_pairs

    def _get_output_headers(self):
        # Transform headers
        output_headers = copy.deepcopy(self.input_headers)

        for x, y in self.feature_pairs:
            feature_out = f"{x}_{y}"
            if (
                x in self.input_headers["categoric"]
                and y in self.input_headers["categoric"]
            ):
                output_headers["categoric"].append(feature_out)
            else:
                output_headers["numeric"].append(feature_out)

        return output_headers

    def _transform(self, matching_data: MatchingData) -> MatchingData:
        # Make a copy of the input data, to which we will add the transformed data
        data = matching_data.copy().data

        # Get rescaled features / We don't want to write these back to the
        # output df, because we only want to add features, not change existing
        # ones. Use another preprocessor if you want to rescale features.
        tmp_data = matching_data.copy().data
        tmp_data.loc[:, self.input_headers["numeric"]] = self.scaler.transform(
            data[self.input_headers["numeric"]]
        )
        for x, y in self.feature_pairs:
            feature_out = f"{x}_{y}"
            data.loc[:, feature_out] = tmp_data[x] * tmp_data[y]

        matching_data = MatchingData(
            data=data,
            headers=self.output_headers,
            population_col=matching_data.population_col,
        )
        return matching_data


class DecisionTreeEncoder(BaseMatchingPreprocessor):
    """
    DecisionTreeEncoder transforms all covariates into binary coviates corresponding
    to their terminal (leaf) position on a decision tree.
    """

    _decision_tree_params = {"criterion": "entropy", "min_samples_leaf": 25}

    def __init__(self, keep_original_features=False, **decision_tree_params):
        self.keep_original_features = keep_original_features
        self._decision_tree_params.update(decision_tree_params)
        super(DecisionTreeEncoder, self).__init__()

    def _fit(self, matching_data: MatchingData) -> None:
        # Set up decision tree
        self._decision_tree_params.setdefault(
            "max_leaf_nodes", len(matching_data.headers["all"])
        )
        self.decision_tree = DecisionTreeClassifier(**self._decision_tree_params)

        # Train decision tree model
        X_train = matching_data.data[matching_data.headers["all"]]
        y_train = matching_data.data[matching_data.population_col]
        self.decision_tree.fit(X_train, y_train)
        self._leaf_features = np.unique(self.decision_tree.apply(X_train)).tolist()

    def _get_output_headers(self):
        # Transform headers
        if self.keep_original_features:
            output_headers = copy.deepcopy(self.input_headers)
        else:
            output_headers = MatchingHeaders(categoric=[], numeric=[])

        for leaf in self._leaf_features:
            output_headers["categoric"].append(f"node_{leaf}")

        return output_headers

    def _transform(self, matching_data: MatchingData) -> MatchingData:
        # Make a copy of the input data, to which we will add the transformed data
        data = matching_data.copy().data

        leaves = self.decision_tree.apply(data[self.input_headers["all"]])
        for leaf in self._leaf_features:
            leaf_term = f"node_{leaf}"
            data.loc[:, leaf_term] = (leaves == leaf).astype(int)

        matching_data = MatchingData(
            data=data,
            headers=self.output_headers,
            population_col=matching_data.population_col,
        )
        return matching_data

    def plot_tree(self):
        figwidth = max(self._decision_tree_params["max_leaf_nodes"], 8)
        plt.figure(figsize=(figwidth, 3 * figwidth / 4))

        tree.plot_tree(
            self.decision_tree,
            feature_names=self.input_headers["all"],
            class_names=self.decision_tree.classes_,
            impurity=False,
            node_ids=True,
        )


class ChainPreprocessor(BaseMatchingPreprocessor):
    """
    ChainPreprocessor applies a sequences of Preprocessors.

    :param preprocessors: A list of preprocessors to be applied in sequence to
        the input data.
    """

    def __init__(self, preprocessors: List[BaseMatchingPreprocessor]):
        self.preprocessors = preprocessors
        super(ChainPreprocessor, self).__init__()

    def _fit(self, matching_data: MatchingData) -> None:
        self.input_headers = matching_data.headers
        for p in self.preprocessors:
            matching_data = p.fit_transform(matching_data)

    def _get_output_headers(self):
        # Output variables determined by last preprocessor
        return self.preprocessors[-1].output_headers

    def _transform(self, matching_data: MatchingData) -> MatchingData:
        for p in self.preprocessors:
            matching_data = p.transform(matching_data)
        return matching_data

    def _get_feature_names_out(self, feature_name_in: str) -> List[str]:
        features_in = [feature_name_in]
        for p in self.preprocessors:
            features_in = p.get_feature_names_out(features_in)
        return features_in


class StandardMatchingPreprocessor(ChainPreprocessor):
    """
    The StandardMatchingPreprocessor implements the "standard" preprocessing
    used in many applications for matching. In particular, this preprocessor
    chains the FloatEncoder and CategoricOneHotEncoder and leaves numeric
    covariates unchanged. This preprocessor can be used to compute the mean
    standardized mean difference.
    """

    def __init__(self, drop: Optional[str] = "first"):
        pp_categoric = CategoricOneHotEncoder(drop=drop)
        pp_basic = FloatEncoder()
        preprocessors = [pp_categoric, pp_basic]
        super(StandardMatchingPreprocessor, self).__init__(preprocessors=preprocessors)


class GammaPreprocessor(ChainPreprocessor):
    """
    A preprocessing chain that can be used to compute distance based on
    cumulative distributions, a statistic that is often referred to as "gamma".
    """

    def __init__(
        self,
        n_bins: int = 10,
        encode: str = "onehot-dense",
        cumulative: bool = True,
        drop: Optional[str] = "first",
    ):
        pp_categoric = CategoricOneHotEncoder(drop=drop)
        pp_numeric = NumericBinsEncoder(
            n_bins=n_bins, encode=encode, cumulative=cumulative
        )
        preprocessors = [pp_categoric, pp_numeric]
        super(GammaPreprocessor, self).__init__(preprocessors=preprocessors)


class BetaXPreprocessor(ChainPreprocessor):
    """
    A preprocessing chain that can be used to compute distance based on
    standardized mean difference on the basic covariates as well as second-order
    cross terms.
    """

    def __init__(
        self,
        drop: Optional[str] = "first",
        max_cross_terms: Optional[int] = None,
    ):
        pp_categoric = CategoricOneHotEncoder(drop=drop)
        pp_cross_terms = CrossTermsPreprocessor(max_cross_terms)
        preprocessors = [pp_cross_terms, pp_categoric]
        super(BetaXPreprocessor, self).__init__(preprocessors=preprocessors)


class GammaXPreprocessor(ChainPreprocessor):
    """
    A preprocessing chain that can be used to compute distance based on
    cumulative distributions including 1D marginal distributions on the basic
    covariates as well as second-order cross terms.
    """

    def __init__(
        self,
        n_bins: int = 10,
        encode: str = "onehot-dense",
        cumulative: bool = True,
        drop: Optional[str] = "first",
        max_cross_terms: Optional[int] = None,
    ):
        pp_categoric = CategoricOneHotEncoder(drop=drop)
        pp_cross_terms = CrossTermsPreprocessor(max_cross_terms)
        pp_numeric = NumericBinsEncoder(
            n_bins=n_bins, encode=encode, cumulative=cumulative
        )
        preprocessors = [pp_cross_terms, pp_categoric, pp_numeric]
        super(GammaXPreprocessor, self).__init__(preprocessors=preprocessors)

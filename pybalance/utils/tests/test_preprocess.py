import pandas as pd
import numpy as np

from pybalance.utils.preprocess import (
    FloatEncoder,
    CategoricOneHotEncoder,
    NumericBinsEncoder,
    DecisionTreeEncoder,
    ChainPreprocessor,
)
from pybalance.sim import generate_toy_dataset
from pybalance.utils import MatchingData, MatchingHeaders


def test_float_encoder():
    N = 100
    np.random.seed(1234)
    data = pd.DataFrame.from_dict(
        {
            "cat1": np.random.choice(list("abcdefghijk"), size=N, replace=True),
            "cat1": np.random.choice([1, 2, 3, 4, 5, 10], size=N, replace=True),
        }
    )
    data.loc[:, "population"] = ["pool"] * len(data)
    data.loc[:, "patient_id"] = list(range(len(data)))
    matching_data = MatchingData(data=data)

    pp_float = FloatEncoder()
    transformed_data = pp_float.fit_transform(matching_data)

    assert (
        transformed_data.headers["categoric"] == transformed_data.headers["categoric"]
    )
    assert transformed_data.headers["numeric"] == transformed_data.headers["numeric"]

    # test that the transformed data has same number of unique values as the
    # original data
    assert data["cat1"].nunique() == data["cat1"].nunique()

    # test that the transformed values align with old values
    combined_data = data.merge(
        transformed_data.data, on="patient_id", suffixes=["_old", "_new"]
    )
    for val, df in combined_data.groupby("cat1_old"):
        assert df["cat1_new"].nunique() == 1


def test_categoric_encoder():
    data = pd.DataFrame.from_dict(
        {"cat1": ["a", "b", "c", "b"], "cat2": [1, 2, 3, 4], "bin1": [0, 1, 0, 1]}
    )
    data.loc[:, "population"] = ["pool"] * len(data)
    data.loc[:, "patient_id"] = list(range(len(data)))
    matching_data = MatchingData(data=data)

    pp_cat = CategoricOneHotEncoder(drop="if_binary")
    transformed_data = pp_cat.fit_transform(matching_data)
    assert len(transformed_data.headers["numeric"]) == len(
        matching_data.headers["numeric"]
    )

    # check correct number of output columns
    n_unique_categories = matching_data[matching_data.headers["categoric"]].nunique()
    n_columns_expected = n_unique_categories[n_unique_categories > 2].sum() + len(
        n_unique_categories[n_unique_categories == 2]
    )
    assert len(transformed_data.headers["categoric"]) == n_columns_expected

    # check all output columns are now binary
    assert all(transformed_data[transformed_data.headers["categoric"]].nunique() == 2)

    # Check actual output values
    # Technically not really a great test, since the column naming might not be stable.
    for cat in matching_data.headers["categoric"]:
        if matching_data[cat].nunique() > 2:
            for val in matching_data[cat].unique():
                assert all(
                    transformed_data[f"{cat}_{val}"] == (data[cat] == val).astype(int)
                )

    # check transforming on data with missing categories still includes missing
    # categories in the output
    data_with_missing_categories = pd.DataFrame.from_dict(
        {"cat1": ["a", "b", "b"], "cat2": [1, 2, 4], "bin1": [0, 1, 1]}
    )
    data_with_missing_categories.loc[:, "population"] = ["pool"] * len(
        data_with_missing_categories
    )
    data_with_missing_categories.loc[:, "patient_id"] = list(
        range(len(data_with_missing_categories))
    )
    matching_data = MatchingData(data=data_with_missing_categories)
    transformed_data = pp_cat.transform(matching_data)
    assert len(transformed_data.headers["categoric"]) == n_columns_expected


def test_categoric_noop():
    # Test when there are no categoric variables that the preprocessor does
    # nothing

    data1 = pd.DataFrame.from_dict({"col": np.linspace(0, 1, 2001)})
    data2 = pd.DataFrame.from_dict({"col": np.linspace(0.5, 1.5, 1001)})
    data1.loc[:, "population"] = "pool"
    data2.loc[:, "population"] = "target"

    matching_data = MatchingData(pd.concat([data1, data2]))
    pp = CategoricOneHotEncoder()
    m_out = pp.fit_transform(matching_data)
    assert np.all(
        m_out[matching_data.headers.all] == matching_data[matching_data.headers.all]
    )


def test_categoric_drop():
    data = pd.DataFrame.from_dict(
        {
            "population": ["a", "b", "a", "b"],
            "gender": ["Male", "Female", "Male", "Male"],
            "country": ["1", "2", "3", "4"],
        }
    )
    m = MatchingData(data)
    pp = CategoricOneHotEncoder(drop="first")
    pp.fit_transform(m)
    assert pp.get_feature_names_out("gender") == ["gender_Male"]
    assert pp.get_feature_names_out("country") == [
        "country_2",
        "country_3",
        "country_4",
    ]


def test_numeric_bins_encoder():
    N = 10
    df = pd.DataFrame.from_dict({"x": list(range(N)), "population": ["x"] * N})
    headers = MatchingHeaders(numeric=["x"], categoric=[])
    m = MatchingData(df, headers=headers)
    p = NumericBinsEncoder(n_bins=N)
    df_transformed = p.fit_transform(m)
    assert df_transformed["x"].nunique() == N
    for i in range(N):
        assert df_transformed.data.loc[i]["x"] == i


def test_numeric_bins_encoder_feature_names():
    # Test number of feature names out returned == number of bins. Since the
    # tested method is based on a regexp, try to use some confusing column
    # names.

    # Start from basic dataset
    m = generate_toy_dataset()
    data = m.data

    # Add features that might confuse the regexp
    data.loc[
        :, "age"
    ] = 1  # This will make the number of returned bins < number requested
    data.loc[:, "age_weight"] = data["age"] * data["weight"]
    data.loc[:, "age*weight"] = (
        data["age"] * data["weight"]
    )  # include special regexp symbol
    data.loc[:, "age-weight"] = data["age"] * data["weight"]
    m = MatchingData(data)

    N = 15
    pp_numeric = NumericBinsEncoder(n_bins=N, cumulative=False, encode="onehot")
    pp_numeric.fit_transform(m)

    for feature_name_in in m.headers.numeric:
        mapped_cols = pp_numeric.get_feature_names_out(feature_name_in)
        feature_index = pp_numeric.discretizer.feature_names_in_.tolist().index(
            feature_name_in
        )
        assert len(mapped_cols) == pp_numeric.discretizer.n_bins_[feature_index]


def test_numeric_bins_encoder_feature_names_order():
    # Test that the order of the feature names out imply increasing values. This
    # is a prerequisite for generating the cumulative bins.

    # Start from basic dataset
    m = generate_toy_dataset()
    data = m.data

    # add a feature that might confuse the order of bins
    data.loc[:, "age"] = -data["age"]
    data.loc[:, "age*weight"] = data["age"] * data["weight"]
    m = MatchingData(data)

    N = 15
    pp_numeric = NumericBinsEncoder(n_bins=N, cumulative=False, encode="onehot")
    m_out = pp_numeric.fit_transform(m)

    for feature_name_in in m.headers.numeric:
        mapped_cols = pp_numeric.get_feature_names_out(feature_name_in)
        for left, right in zip(mapped_cols[:-1], mapped_cols[1:]):
            left_max = m.data.loc[m_out[left] == 1, feature_name_in].max()
            right_min = m.data.loc[m_out[right] == 1, feature_name_in].min()
            assert left_max < right_min


def test_numeric_bins_encoder_onehot_cumulative():
    # Test the cumulative bins are generated correctly.

    # Start from basic dataset
    m = generate_toy_dataset()
    data = m.data

    # add a feature that might confuse the order of bins
    data.loc[:, "age"] = -data["age"]
    data.loc[:, "age*weight"] = data["age"] * data["weight"]
    m = MatchingData(data)

    N = 15
    pp_numeric = NumericBinsEncoder(n_bins=N, cumulative=True, encode="onehot")
    m_out = pp_numeric.fit_transform(m)

    # test that the ORDER of the feature names out imply increasing values
    for feature_name_in in m.headers.numeric:
        mapped_cols = pp_numeric.get_feature_names_out(feature_name_in)
        for left, right in zip(mapped_cols[:-1], mapped_cols[1:]):
            # test that left == 1 --> right == 1
            right_min = m_out.data.loc[m_out[left] == 1, right].min()
            assert right_min == 1


def test_numeric_bins_encoder_onehot_cumulative2():
    data = pd.DataFrame.from_dict(
        {"age": [1, 2, 3, 4, 5, 6.0, 7, 8, 9, 10, 11, 2, 3, 4]}
    )
    data.loc[:, "population"] = ["a"] * len(data["age"])
    m = MatchingData(data, MatchingHeaders(numeric=["age"], categoric=[]))

    pp = NumericBinsEncoder(cumulative=True, n_bins=10, strategy="quantile")
    pp.fit(m)
    m_out = pp.transform(m)

    cols_out = pp.get_feature_names_out("age")
    # check last column is dropped
    assert m_out.data.loc[m_out["age"] == 10, cols_out].values.max() == 0
    assert m_out.data.loc[m_out["age"] == 1, cols_out].values.min() == 1


def test_decision_tree_encoder():
    m = generate_toy_dataset()
    pp_float = DecisionTreeEncoder(max_leaf_nodes=3)
    m_out = pp_float.fit_transform(m)
    assert len(m_out.headers.all) == 3

    pp_float = DecisionTreeEncoder(keep_original_features=True, max_leaf_nodes=3)
    m_out = pp_float.fit_transform(m)
    assert len(m_out.headers.all) == 3 + len(m.headers.all)


def test_chain_preprocessor_output_names():
    m = generate_toy_dataset()
    pp_n = NumericBinsEncoder(encode="onehot", n_bins=6)
    pp_c = CategoricOneHotEncoder(drop=None)
    pp = ChainPreprocessor([pp_c, pp_n])
    pp.fit(m)

    expected = {
        "gender": ["gender_0.0", "gender_1.0"],
        "binary_0": ["binary_0_0", "binary_0_1"],
        "binary_1": ["binary_1_0", "binary_1_1"],
        "binary_2": ["binary_2_0", "binary_2_1"],
        "binary_3": ["binary_3_0", "binary_3_1"],
        "country": [
            "country_0",
            "country_1",
            "country_2",
            "country_3",
            "country_4",
            "country_5",
        ],
        "haircolor": ["haircolor_0", "haircolor_1", "haircolor_2"],
        "age": ["age_0.0", "age_1.0", "age_2.0", "age_3.0", "age_4.0", "age_5.0"],
        "weight": [
            "weight_0.0",
            "weight_1.0",
            "weight_2.0",
            "weight_3.0",
            "weight_4.0",
            "weight_5.0",
        ],
        "height": [
            "height_0.0",
            "height_1.0",
            "height_2.0",
            "height_3.0",
            "height_4.0",
            "height_5.0",
        ],
    }

    for c in m.headers["all"]:
        assert pp.get_feature_names_out(c) == expected[c]


def test_feature_names_out():
    pp = NumericBinsEncoder(cumulative=True, n_bins=5)
    m = generate_toy_dataset()
    pp.fit_transform(m)
    assert set(pp.get_feature_names_out()) == set(pp.output_headers["all"])

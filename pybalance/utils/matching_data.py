from __future__ import annotations
from typing import List, Union, Dict, Optional
import copy
from dataclasses import dataclass
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class MatchingHeaders(object):
    """
    MatchingHeaders is a simple data structure to store information about which
    features to be used for matching and separating these features into
    categoric (e.g. country, gender) and numeric (e.g. age, weight) types.

    :param categoric: List of features to be treated as categoric variables.
    :param numeric:  List of features to be treated as numeric variables.
    """

    categoric: List[str]
    numeric: List[str]

    @property
    def all(self):
        return self.categoric + self.numeric

    # for backward compatability
    def __getitem__(self, key):
        return {"numeric": self.numeric, "categoric": self.categoric, "all": self.all}[
            key
        ]


def infer_matching_headers(
    data: pd.DataFrame,
    max_categories: int = 10,
    ignore_cols: List[str] = ["patient_id", "patientid", "population", "index_date"],
) -> MatchingHeaders:
    """
    This utility function guesses which columns are numeric and which columns
    are categoric from input data. The data can be passed either as separate
    data frames target and pool or combined in one and passed with keyword
    argument data. The function returns a dictionary with keys 'numeric',
    'categoric' and 'all' with values equal to the list of column names of the
    given type. By default, the function ignores patient_id and population
    columns.
    """
    usecols = [c for c in data.columns if c not in ignore_cols]
    data = data[usecols]

    categoric_cols = data.columns[data.nunique() <= max_categories].values.tolist()
    numeric_cols = []
    proposed_numeric = data.columns[data.nunique() > max_categories].values.tolist()
    for col in proposed_numeric:
        try:
            data[col].astype(float)
        except ValueError:
            # If column cannot be cast to numeric, treat as categoric
            logger.warning(
                f"Unable to cast {col} to float. Treating as categoric with {data[col].nunique()} categories."
            )
            categoric_cols.append(col)
        else:
            numeric_cols.append(col)

    headers = MatchingHeaders(numeric=numeric_cols, categoric=categoric_cols)

    logger.debug(f"Inferred headers: {headers}")

    return headers


def _make_quantile_function(q):
    def f(x):
        return x.quantile(q)

    if q == 0:
        name = "min"
    elif q == 1:
        name = "max"
    elif q == 0.5:
        name = "median"
    else:
        name = f"q{int(100 * q)}"
    f.__name__ = name
    return f


def _load_matching_data(path):
    if path.endswith(".csv") or path.endswith("csv.gz"):
        data = pd.read_csv(path)
    elif path.endswith(".parquet"):
        data = pd.read_parquet(path)
    else:
        raise ValueError(f"Unknown file format: {path}.")
    return data


class MatchingData(object):
    """
    It is common in matching problems to require basic metadata about the data
    in order to perform matching. For instance, the data may contain columns
    such as "patient_id", "population" and "index_date", which are not intended
    to be used for matching but which must "go along for the ride" and follow
    the main data everywhere. MatchingData is a wrapper around pandas.DataFrame
    that includes this additional required logic about the columns. Features
    required for matching are described by a "headers" field, while other
    columns exist alongside. See MatchingHeaders.

    :param data: Data frame containing both matching feature data for all
        populations as well as at least one additional column specifying to
        which population each row belongs. If a string is passed, it is assumed
        to be a path to the data frame.

    :param headers: A MatchingHeaders object with keys "numeric" and "categoric"
        and whoses values are names of columns to be used for matching. If None
        is passed, headers will be inferred based on how many unique values each
        column has. As guessing the headers can lead to errors, it is
        recommended to supply them explicitly.

    :param population_col: Name of the column used to split data into
        subpopulations.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        headers: Optional[MatchingHeaders] = None,
        population_col: str = "population",
    ):
        # load data frames if paths are passed
        if isinstance(data, str):
            data = _load_matching_data(data)
        self._data = data
        self.population_col = population_col
        self._set_headers(headers)

        if population_col not in data.columns:
            raise KeyError(
                f"""
            Cannot split into populations based on {population_col}. Column not
            present in data frame.
            """
            )

    def _set_headers(
        self, headers: Optional[Union[Dict[str, List[str]], MatchingHeaders]]
    ) -> None:
        """
        Set private data needed to construct the headers property. Headers are
        set at initialization and are considered immutable. If you need to
        change the headers, you must create a new instance of MatchingData.
        """
        if headers is None:
            headers = infer_matching_headers(
                self.data,
                ignore_cols=[
                    "patient_id",
                    "patientid",
                    "index_date",
                    self.population_col,
                ],
            )
        elif not isinstance(headers, MatchingHeaders):
            headers = MatchingHeaders(**headers)

        self.headers = headers

    def get_population(self, population: str) -> pd.DataFrame:
        """
        Get the matching data for a population by its name.
        """
        pop = self._data[self._data[self.population_col] == population]
        if not len(pop):
            raise KeyError(f"Population {population} not found!")
        return pop

    @property
    def populations(self) -> List[str]:
        """
        List of all populations present in the MatchingData object.
        """
        return sorted(self[self.population_col].unique().tolist())

    @property
    def data(self) -> pd.DataFrame:
        """
        Pointer to underlying pandas DataFrame.
        """
        # Defining data as a property prevents the user from manually setting
        # the data.
        return self._data

    def sample(self, n: int = 5) -> pd.DataFrame:
        """
        Sample underlying pandas DataFrame.
        """
        return self.data.sample(n=n)

    def head(self, n: int = 5) -> pd.DataFrame:
        """
        Return first n rows from underlying pandas DataFrame.
        """
        return self.data.head(n=n)

    def tail(self, n: int = 5) -> pd.DataFrame:
        """
        Return last n rows from underlying pandas DataFrame.
        """
        return self.data.tail(n=n)

    def __getitem__(self, key: Union[List[str], str]):
        return self.data.__getitem__(key)

    def copy(self) -> MatchingData:
        """
        Create a new MatchingData instance with exact same data and metadata.
        """
        data = copy.deepcopy(self.data)
        headers = copy.deepcopy(self.headers)
        matching_data = MatchingData(
            data=data, headers=headers, population_col=self.population_col
        )
        return matching_data

    def append(self, df: pd.DataFrame, name: Optional[str] = None) -> None:
        """
        Append a population to an existing MatchingData instance. This operation
        is inplace.
        """
        if not set([self.population_col] + self.headers["all"]) <= set(df.columns):
            missing_columns = set([self.population_col] + self.headers["all"]) - set(
                df.columns
            )
            raise ValueError(
                f"Required columns {list(missing_columns)} are missing from data to be appended."
            )

        # copy input data to avoid undesired side effects
        df = copy.deepcopy(df)
        if name is not None:
            df.loc[:, self.population_col] = name

        self._data = pd.concat([self._data, df])

    def to_csv(self, *args, **kwargs):
        """
        Write underlying pandas DataFrame to csv. Call signature is identical to
        pandas method.
        """
        return self.data.to_csv(*args, **kwargs)

    def to_parquet(self, *args, **kwargs):
        """
        Write underlying pandas DataFrame to parquet. Call signature is
        identical to pandas method.
        """
        return self.data.to_parquet(*args, **kwargs)

    def __str__(self):
        return f"""
Headers Numeric:
{self.headers['numeric']}

Headers Categoric:
{self.headers['categoric']}

Populations:
{self.populations}

{self.data}"""

    def _repr_html_headers_(self):
        return f"""
        <b>Headers Numeric: </b><br>
        {self.headers['numeric']}<br><br>
        <b>Headers Categoric: </b><br>
        {self.headers['categoric']} <br><br>
        <b>Populations</b> <br>
        {self.populations} <br>
        """

    def _repr_html_(self):
        return self._repr_html_headers_() + self.data._repr_html_()

    def __len__(self):
        return len(self.data)

    def counts(self):
        counts = self.data.reset_index().groupby(self.population_col).count()[["index"]]
        counts.columns = ["N"]
        return counts

    def describe_numeric(
        self,
        aggregations=["mean", "std"],
        quantiles=[0, 0.25, 0.5, 0.75, 1],
        long_format=True,
    ) -> pd.DataFrame:
        """
        Create a summary statistics table split by population for numeric variables.
        """
        # numeric
        aggregations = aggregations + [_make_quantile_function(q) for q in quantiles]
        agg = dict((c, aggregations) for c in self.headers["numeric"])
        agg = self.data.reset_index().groupby(self.population_col).agg(agg).T
        agg.columns = [c for c in agg.columns]
        agg = agg.round(decimals=2)

        if not long_format:
            agg = agg.unstack(level=-1)

        return agg

    def describe_categoric(self, normalize=True) -> pd.DataFrame:
        """
        Create a summary statistics table split by population for categoric variables.
        """
        counts = self.counts()["N"]
        counts = pd.DataFrame.from_records(
            [counts.values.astype(int).tolist()],
            index=pd.MultiIndex.from_tuples([(f"{self.population_col} size", "N")]),
            columns=counts.index.values.tolist(),
        )

        # categoric
        out = [counts]
        for cat in self.headers["categoric"]:
            tmp = (
                self.data.reset_index()
                .groupby(["population", cat])
                .count()[["index"]]
                .reset_index()
            )
            tmp.loc[:, "feature"] = cat
            tmp = tmp.pivot(
                index=["feature", cat], columns=["population"], values=["index"]
            )
            tmp.columns = [c[1] for c in tmp.columns]
            tmp.index.names = ["feature", "value"]
            out.append(tmp)

        out = pd.concat(out).fillna(0)

        if normalize:
            for c in counts.columns:
                out.loc[:, c] = out[c] / counts.iloc[0][c]
        else:
            out = out.astype(int)

        return out

    def describe(
        self,
        normalize: bool = True,
        aggregations: List[str] = ["mean", "std"],
        quantiles: List[float] = [0, 0.25, 0.5, 0.75, 1],
    ) -> pd.DataFrame:
        """
        Calls describe_categoric() and describe_numeric() and returns the
        results in a single dataframe.
        """
        c = self.describe_categoric(normalize)
        n = self.describe_numeric(aggregations, quantiles)
        return pd.concat([c, n])


def split_target_pool(
    matching_data: MatchingData,
    pool_name: Optional[str] = None,
    target_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Split matching_data into target and pool populations based. If
    the names of the target and pool populations are not
    explicitly provided, the routine will attempt to infer their names,
    assuming that the target population is the smaller population.
    """

    if isinstance(target_name, str) and isinstance(pool_name, str):
        target = matching_data.get_population(target_name)
        pool = matching_data.get_population(pool_name)
    elif isinstance(target_name, str) or isinstance(pool_name, str):
        if len(matching_data.populations) != 2:
            raise ValueError(
                f"""
            Cannot split into exactly two populations based on {matching_data.population_col}.
            Found populations: {','.join(matching_data.populations)}.
            """
            )
        if isinstance(target_name, str):
            pool_name = [p for p in matching_data.populations if p != target_name][0]
        if isinstance(pool_name, str):
            target_name = [p for p in matching_data.populations if p != pool_name][0]
        target = matching_data.get_population(target_name)
        pool = matching_data.get_population(pool_name)
    else:
        if len(matching_data.populations) != 2:
            raise ValueError(
                f"""
            Cannot split into exactly two populations based on {matching_data.population_col}.
            Found populations: {','.join(matching_data.populations)}.
            """
            )
        inferred_pool_name = matching_data.populations[0]
        inferred_target_name = matching_data.populations[1]
        target = matching_data.get_population(inferred_target_name)
        pool = matching_data.get_population(inferred_pool_name)

        # bigger population considered pool, just a convention, no real effect
        if len(pool) < len(target):
            _pool = pool
            pool = target
            target = _pool

    return target, pool

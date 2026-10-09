from __future__ import annotations
from typing import Any, List, Union, Dict, Optional
import copy
from dataclasses import dataclass, field
from pathlib import Path
import math
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


def _normalize_numeric_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    extra = set(stats) - {"mean", "std", "median", "quantile", "min", "max"}
    if extra:
        raise ValueError(
            f"Unknown numeric aggregate stat key(s) {sorted(extra)}. Got: {stats}"
        )
    out: Dict[str, Any] = {}
    if "mean" in stats:
        out["mean"] = float(stats["mean"])
    if "std" in stats and stats["std"] is not None:
        out["std"] = float(stats["std"])

    # A disclosed quantile (q, value) means P(raw <= value) = q for the raw
    # column; it is matched by dichotomizing the raw column at `value` and
    # constraining the resulting indicator's rate to 1 - q. 'median' is just
    # sugar for the single quantile (0.5, value), and 'min'/'max' for the 0th and
    # 100th quantile. Quantile constraints are independent of, and may be
    # disclosed alongside, 'mean'/'std' on the same raw column. See
    # AggregateTargetBalanceCalculator / FixedNumericBinsEncoder.
    quantiles = []
    if "min" in stats:
        quantiles.append((0.0, float(stats["min"])))
    if "median" in stats:
        quantiles.append((0.5, float(stats["median"])))
    if "max" in stats:
        quantiles.append((1.0, float(stats["max"])))
    if "quantile" in stats:
        for q, value in stats["quantile"]:
            q = float(q)
            if not 0.0 <= q <= 1.0:
                raise ValueError(
                    f"Quantile proportion must be between 0 and 1; got {q}."
                )
            quantiles.append((q, float(value)))
    if quantiles:
        out["quantile"] = sorted(quantiles)

    if not out:
        raise ValueError(
            "Numeric aggregate stats must include at least one of 'mean', 'std', "
            f"'median', 'min', 'max' or 'quantile'. Got: {stats}"
        )
    return out


def _normalize_categoric_rates(rates: Dict[Any, Any], feature: str) -> Dict[Any, float]:
    if not rates:
        raise ValueError(f"Categoric feature '{feature}' has empty rate dictionary.")
    out = {k: float(v) for k, v in rates.items()}
    for category, rate in out.items():
        if not 0.0 < rate < 1.0:
            raise ValueError(
                f"Categoric rate for '{feature}'='{category}' must be strictly "
                f"between 0 and 1; got {rate}."
            )
    # Categories are allowed to be partially disclosed -- e.g. a published
    # Table 1 may give "country: US 60%" without breaking out every other
    # country -- in which case the rest are left unconstrained. Only the
    # impossible case (rates summing to more than 1) is rejected.
    total = sum(out.values())
    if total > 1.0 + 1e-3:
        raise ValueError(
            f"Categoric rates for '{feature}' must not sum to more than 1.0 "
            f"(got {total})."
        )
    return out


# Moment types supported by the long-format moments file (see load_target_moments).
SUPPORTED_MOMENTS = ("mean", "std")


def load_target_moments(path: Union[str, Path]) -> Dict[str, Dict[str, float]]:
    """
    Load a long-format "moments" file describing an aggregate target into a
    dict of variable -> {moment: value}.

    The file must have columns ``variable``, ``moment``, ``value``, one row
    per disclosed statistic, e.g.::

        variable,moment,value
        AGE,mean,68.0
        PSA,mean,65.0
        PSA,std,36.0
        ECOG_0,mean,0.515

    Only ``moment`` values in :data:`SUPPORTED_MOMENTS` (currently "mean" and
    "std") are recognized; unsupported moments and missing (NaN) values are
    skipped with a warning so the file can carry extra disclosed statistics
    without breaking matching.
    """
    df = pd.read_csv(path)
    required = {"variable", "moment", "value"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Moments file {path} missing required columns: {sorted(missing_cols)}"
        )

    moments: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        variable, moment, value = row["variable"], row["moment"], row["value"]
        if moment not in SUPPORTED_MOMENTS:
            logger.warning(
                f"Ignoring unsupported moment '{moment}' for variable '{variable}' in {path}."
            )
            continue
        if pd.isna(value):
            continue
        moments.setdefault(variable, {})[moment] = float(value)
    return moments


AGGREGATE_TARGET_CSV_COLUMNS = ("feature", "statistic", "parameter", "value")
_NUMERIC_CSV_STATISTICS = ("mean", "std", "median", "min", "max")


def _parse_level(text: str) -> Any:
    # Category levels are stored as text; recover int/float levels where possible.
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def _csv_float(text: str, what: str, line: int) -> float:
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"Line {line}: {what} must be a number; got {text!r}.")


@dataclass
class AggregateTarget:
    """
    Published / summary-only target population for matching.

    Use when patient-level rows are available for the pool, but the reference
    population is known only through aggregate statistics (e.g. Table 1 means
    and category prevalences).

    :param n: Sample size of the target population.
    :param numeric: Mapping feature -> any non-empty subset of {"mean",
        "std", "median", "quantile", "min", "max"}. Each disclosed statistic is
        its own constraint; undisclosed ones are not constrained. ``median`` and
        ``quantile`` describe disclosed quantiles of the raw column (e.g. a
        published median, or "80% of patients weigh more than 70kg" as
        ``{"quantile": [(0.2, 70)]}``). A ``std`` without a ``mean`` is the
        spread around the matched subset's own mean. ``min`` and ``max`` are the
        0th and 100th quantile; like every other constraint they are *soft*:
        the matcher penalizes the fraction of patients below ``min`` / above
        ``max`` rather than forbidding them. (``min`` is exclusive: a patient
        exactly at ``min`` counts as below it.)
    :param categoric: Mapping feature -> {category: rate}. Categories may be
        partially disclosed -- e.g. a published Table 1 may give only
        "country: US 60%" -- in which case rates need not sum to 1 and
        unlisted categories get no constraint at all.
    :param headers: Optional explicit MatchingHeaders. If omitted, inferred from
        the keys of ``numeric`` and ``categoric``.
    """

    n: int
    numeric: Dict[str, Dict[str, float]] = field(default_factory=dict)
    categoric: Dict[str, Dict[Any, float]] = field(default_factory=dict)
    headers: Optional[MatchingHeaders] = None

    def __post_init__(self) -> None:
        if self.n is None or int(self.n) <= 0:
            raise ValueError("AggregateTarget.n must be a positive integer.")
        self.n = int(self.n)

        self.numeric = {
            feature: _normalize_numeric_stats(stats)
            for feature, stats in self.numeric.items()
        }
        self.categoric = {
            feature: _normalize_categoric_rates(rates, feature)
            for feature, rates in self.categoric.items()
        }

        if self.headers is None:
            self.headers = MatchingHeaders(
                numeric=list(self.numeric.keys()),
                categoric=list(self.categoric.keys()),
            )
        elif not isinstance(self.headers, MatchingHeaders):
            self.headers = MatchingHeaders(**self.headers)

        numeric_keys = set(self.numeric)
        categoric_keys = set(self.categoric)
        if numeric_keys & categoric_keys:
            raise ValueError(
                "Features cannot be both numeric and categoric in AggregateTarget: "
                f"{sorted(numeric_keys & categoric_keys)}"
            )
        if set(self.headers.numeric) != numeric_keys:
            raise ValueError(
                "headers.numeric must match numeric feature keys. "
                f"headers={self.headers.numeric}, numeric={sorted(numeric_keys)}"
            )
        if set(self.headers.categoric) != categoric_keys:
            raise ValueError(
                "headers.categoric must match categoric feature keys. "
                f"headers={self.headers.categoric}, categoric={sorted(categoric_keys)}"
            )

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AggregateTarget":
        """
        Build an AggregateTarget from a plain dictionary.

        Expected shape::

            {
                "n": 200,
                "numeric": {"age": {"mean": 65.2, "std": 10.1}, ...},
                "categoric": {"sex": {"F": 0.45, "M": 0.55}, ...},
                # optional:
                "headers": {"numeric": [...], "categoric": [...]},
            }
        """
        if "n" not in payload:
            raise ValueError("AggregateTarget.from_dict requires key 'n'.")
        headers = payload.get("headers")
        if headers is not None and not isinstance(headers, MatchingHeaders):
            headers = MatchingHeaders(**headers)
        return cls(
            n=payload["n"],
            numeric=dict(payload.get("numeric", {})),
            categoric=dict(payload.get("categoric", {})),
            headers=headers,
        )

    @classmethod
    def from_csv(cls, path: Union[str, Path]) -> "AggregateTarget":
        """
        Load an AggregateTarget from a CSV file in pybalance's standard
        aggregate-target format (the inverse of :meth:`to_csv`).

        The file is in long format, one disclosed statistic per row, with
        columns ``feature``, ``statistic``, ``parameter`` and ``value``::

            feature,statistic,parameter,value
            ,n,,100
            age,mean,,52.3
            age,std,,11.8
            age,median,,51.0
            age,max,,75.0
            weight,quantile,0.2,70.0
            gender,rate,0,0.45
            gender,rate,1,0.55

        - ``n`` (blank ``feature``, blank ``parameter``): target sample size.
          Exactly one row is required.
        - ``mean``, ``std``, ``median``, ``min``, ``max`` (blank
          ``parameter``): numeric statistics of ``feature``. ``min`` and
          ``max`` are soft limits (the 0th and 100th quantile).
        - ``quantile``: ``parameter`` is the proportion ``q`` in [0, 1] and
          ``value`` the cutpoint, i.e. P(feature <= value) = q.
        - ``rate``: ``parameter`` is a category level of ``feature`` and
          ``value`` its rate. Levels may be omitted (partially disclosed).

        A feature is numeric if it has numeric statistics and categoric if it
        has ``rate`` rows. Feature order follows first appearance in the file.
        See :class:`AggregateTarget` for the semantics of each statistic.
        """
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        missing_cols = set(AGGREGATE_TARGET_CSV_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Aggregate target file {path} missing required columns: "
                f"{sorted(missing_cols)}"
            )

        n = None
        numeric: Dict[str, Dict[str, Any]] = {}
        categoric: Dict[str, Dict[Any, float]] = {}
        for i, row in enumerate(df.to_dict("records")):
            line = i + 2  # 1-based, after the header row
            feature = row["feature"].strip()
            statistic = row["statistic"].strip()
            parameter = row["parameter"].strip()
            value = _csv_float(row["value"].strip(), "value", line)

            if statistic == "n":
                if n is not None:
                    raise ValueError(f"Line {line}: duplicate 'n' row.")
                if value != int(value):
                    raise ValueError(f"Line {line}: n must be an integer; got {value}.")
                n = int(value)
                continue

            if not feature:
                raise ValueError(
                    f"Line {line}: statistic '{statistic}' requires a feature."
                )

            if statistic in _NUMERIC_CSV_STATISTICS:
                stats = numeric.setdefault(feature, {})
                if statistic in stats:
                    raise ValueError(
                        f"Line {line}: duplicate '{statistic}' for feature '{feature}'."
                    )
                stats[statistic] = value
            elif statistic == "quantile":
                q = _csv_float(parameter, "quantile parameter", line)
                numeric.setdefault(feature, {}).setdefault("quantile", []).append(
                    (q, value)
                )
            elif statistic == "rate":
                if parameter == "":
                    raise ValueError(
                        f"Line {line}: 'rate' requires a level in 'parameter'."
                    )
                rates = categoric.setdefault(feature, {})
                level = _parse_level(parameter)
                if level in rates:
                    raise ValueError(
                        f"Line {line}: duplicate rate for '{feature}'='{parameter}'."
                    )
                rates[level] = value
            else:
                raise ValueError(
                    f"Line {line}: unknown statistic '{statistic}'. Expected one of "
                    "n, mean, std, median, min, max, quantile, rate."
                )

        if n is None:
            raise ValueError(f"Aggregate target file {path} has no 'n' row.")
        return cls(n=n, numeric=numeric, categoric=categoric)

    def to_csv(self, path: Union[str, Path, None] = None) -> Optional[str]:
        """
        Write this target in the standard CSV format read by :meth:`from_csv`.
        Returns the CSV text if ``path`` is None, otherwise writes the file.
        """

        def py(x):
            return x.item() if hasattr(x, "item") else x

        rows = [("", "n", "", self.n)]
        for feature in self.headers.numeric:
            stats = self.numeric[feature]
            for statistic in ("mean", "std"):
                if statistic in stats:
                    rows.append((feature, statistic, "", stats[statistic]))
            for q, value in stats.get("quantile", []):
                named = {0.0: "min", 0.5: "median", 1.0: "max"}
                if q in named:
                    rows.append((feature, named[q], "", value))
                else:
                    rows.append((feature, "quantile", q, value))
        for feature in self.headers.categoric:
            for level, rate in self.categoric[feature].items():
                rows.append((feature, "rate", py(level), rate))

        # object dtype keeps n as "100" rather than "100.0" next to float values
        df = pd.DataFrame(
            rows, columns=list(AGGREGATE_TARGET_CSV_COLUMNS), dtype=object
        )
        return df.to_csv(path, index=False)

    @classmethod
    def from_moments_csv(
        cls,
        path: Union[str, Path],
        n: int,
        matching_headers: MatchingHeaders,
        impute_missing_std: Optional[str] = None,
    ) -> "AggregateTarget":
        """
        Build an AggregateTarget from a long-format moments file (see
        :func:`load_target_moments`).

        ``matching_headers`` says which of the pool's matching features are
        numeric vs categoric; only those features are considered, and only
        the features actually present in the file are used to constrain the
        target. Categoric features are expected to be 0/1-coded in the pool;
        their "mean" moment is read as the rate of the "1" level, with the
        complementary rate for "0" filled in automatically.

        This is deliberately forgiving: features in ``matching_headers`` but
        missing from the file are left unconstrained (warning, no error);
        variables in the file but absent from ``matching_headers`` are
        ignored (warning, no error).

        :param impute_missing_std: A published Table 1 frequently discloses a
            mean/median but not a std. By default (None), such a numeric
            feature's std is left unconstrained. Passing "zero" or "mean"
            instead fills in an assumed std for every numeric feature whose
            mean is disclosed but whose std is not (features with an actually
            disclosed std are never touched):

              - "zero": assume minimal variance (the most conservative,
                tightest-possible assumption).
              - "mean": assume std equal to the mean, i.e. coefficient of
                variation 1 -- a crude, high-end guess for the kind of
                right-skewed lab values common in a Table 1.

            Running the same match under both is a simple way to gauge how
            sensitive the result is to the disclosed-variance assumption
            when the trial itself does not report one.
        """
        if impute_missing_std not in (None, "zero", "mean"):
            raise ValueError(
                f"impute_missing_std must be one of None, 'zero', 'mean'; got {impute_missing_std!r}."
            )

        moments = load_target_moments(path)

        extra = set(moments) - set(matching_headers.all)
        for variable in sorted(extra):
            logger.warning(
                f"Ignoring variable '{variable}' in {path}: not in matching_headers."
            )

        numeric = {}
        for feature in matching_headers.numeric:
            stats = moments.get(feature)
            if stats is None or "mean" not in stats:
                logger.warning(
                    f"No target moments for numeric feature '{feature}' in {path}; "
                    "treating as unconstrained."
                )
                continue
            stats = dict(stats)
            if "std" not in stats and impute_missing_std is not None:
                if impute_missing_std == "zero":
                    stats["std"] = 0.0
                else:
                    stats["std"] = abs(stats["mean"])
            numeric[feature] = stats

        categoric = {}
        for feature in matching_headers.categoric:
            stats = moments.get(feature)
            if stats is None or "mean" not in stats:
                logger.warning(
                    f"No target moments for categoric feature '{feature}' in {path}; "
                    "treating as unconstrained."
                )
                continue
            rate = stats["mean"]
            categoric[feature] = {1: rate, 0: 1.0 - rate}

        return cls(n=n, numeric=numeric, categoric=categoric)

    @property
    def feature_names(self) -> List[str]:
        return self.headers.all

    def to_frame(self) -> pd.DataFrame:
        """
        Return this target's disclosed stats as a tidy DataFrame with columns
        feature, type, stat, value. For numeric features "stat" is a moment
        name (e.g. "mean", "std"); for categoric features it is a level.
        """
        rows = []
        for feature, stats in self.numeric.items():
            for stat, value in stats.items():
                if stat == "quantile":
                    for q, cutpoint in value:
                        rows.append(
                            {
                                "feature": feature,
                                "type": "numeric",
                                "stat": {0.0: "min", 1.0: "max"}.get(
                                    q, f"quantile_{q}"
                                ),
                                "value": cutpoint,
                            }
                        )
                    continue
                rows.append(
                    {
                        "feature": feature,
                        "type": "numeric",
                        "stat": stat,
                        "value": value,
                    }
                )
        for feature, rates in self.categoric.items():
            for level, value in rates.items():
                rows.append(
                    {
                        "feature": feature,
                        "type": "categoric",
                        "stat": level,
                        "value": value,
                    }
                )
        return pd.DataFrame(rows, columns=["feature", "type", "stat", "value"])

    def __repr__(self) -> str:
        header = f"AggregateTarget(n={self.n})"
        if self.numeric or self.categoric:
            body = self.to_frame().to_string(index=False)
            return f"{header}\n{body}"
        return header

    def _repr_html_(self) -> str:
        header = f"<b>AggregateTarget</b> (n={self.n})<br>"
        if self.numeric or self.categoric:
            return header + self.to_frame().to_html(index=False)
        return header


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

    Construction patterns::

        # Combined patient-level table with a population column
        MatchingData(df)
        MatchingData(data=df, population_col="population")

        # Explicit patient-level pool and target
        MatchingData(pool=pool_df, target=target_df)

        # Patient-level pool and aggregate-only target
        MatchingData(pool=pool_df, target=AggregateTarget.from_dict(...))

    :param data: Data frame containing both matching feature data for all
        populations as well as at least one additional column specifying to
        which population each row belongs. If a string is passed, it is assumed
        to be a path to the data frame. Mutually exclusive with ``pool`` /
        ``target``.

    :param headers: A MatchingHeaders object with keys "numeric" and "categoric"
        and whoses values are names of columns to be used for matching. If None
        is passed, headers will be inferred based on how many unique values each
        column has (or from ``AggregateTarget`` when the target is aggregate).
        As guessing the headers can lead to errors, it is recommended to supply
        them explicitly.

    :param population_col: Name of the column used to split data into
        subpopulations.

    :param pool: Patient-level pool population. Used with ``target``.

    :param target: Either a patient-level target DataFrame or an
        ``AggregateTarget`` summary. Used with ``pool``.

    :param pool_name: Population label for ``pool`` when using the explicit
        split constructor.

    :param target_name: Population label for ``target`` when using the explicit
        split constructor.
    """

    def __init__(
        self,
        data: Optional[Union[pd.DataFrame, str]] = None,
        headers: Optional[MatchingHeaders] = None,
        population_col: str = "population",
        pool: Optional[Union[pd.DataFrame, str]] = None,
        target: Optional[Union[pd.DataFrame, str, AggregateTarget]] = None,
        pool_name: str = "pool",
        target_name: str = "target",
    ):
        self.population_col = population_col
        self.pool_name = pool_name
        self.target_name = target_name
        self.aggregate_target: Optional[AggregateTarget] = None

        if data is not None and (pool is not None or target is not None):
            raise ValueError("Pass either `data` or `pool`/`target`, not both.")
        if data is None and (pool is None or target is None):
            raise ValueError(
                "MatchingData requires either `data` or both `pool` and `target`."
            )
        if (pool is None) ^ (target is None):
            raise ValueError(
                "When using the explicit split constructor, both `pool` and "
                "`target` are required."
            )

        if data is not None:
            if isinstance(data, str):
                data = _load_matching_data(data)
            self._data = data
            self._set_headers(headers)
            if population_col not in self._data.columns:
                raise KeyError(f"""
            Cannot split into populations based on {population_col}. Column not
            present in data frame.
            """)
            return

        pool_df = _load_matching_data(pool) if isinstance(pool, str) else pool.copy()
        if isinstance(target, AggregateTarget):
            self._init_from_pool_and_aggregate(
                pool_df=pool_df,
                aggregate_target=target,
                headers=headers,
            )
        else:
            target_df = (
                _load_matching_data(target)
                if isinstance(target, str)
                else target.copy()
            )
            self._init_from_pool_and_target_frames(
                pool_df=pool_df,
                target_df=target_df,
                headers=headers,
            )

    def _init_from_pool_and_target_frames(
        self,
        pool_df: pd.DataFrame,
        target_df: pd.DataFrame,
        headers: Optional[MatchingHeaders],
    ) -> None:
        pool_df = pool_df.copy()
        target_df = target_df.copy()
        pool_df.loc[:, self.population_col] = self.pool_name
        target_df.loc[:, self.population_col] = self.target_name
        self._data = pd.concat([pool_df, target_df], ignore_index=True)
        self.aggregate_target = None
        self._set_headers(headers)

    def _init_from_pool_and_aggregate(
        self,
        pool_df: pd.DataFrame,
        aggregate_target: AggregateTarget,
        headers: Optional[MatchingHeaders],
    ) -> None:
        pool_df = pool_df.copy()
        pool_df.loc[:, self.population_col] = self.pool_name
        self._data = pool_df
        self.aggregate_target = aggregate_target

        if headers is None:
            headers = aggregate_target.headers
        self._set_headers(headers)
        self._validate_aggregate_target_features()

    def _validate_aggregate_target_features(self) -> None:
        missing = set(self.aggregate_target.feature_names) - set(self._data.columns)
        if missing:
            raise ValueError(
                "AggregateTarget features missing from pool data: " f"{sorted(missing)}"
            )
        header_features = set(self.headers.all)
        aggregate_features = set(self.aggregate_target.feature_names)
        if not aggregate_features <= header_features:
            raise ValueError(
                "AggregateTarget features must be included in MatchingData headers. "
                f"Extra features: {sorted(aggregate_features - header_features)}"
            )

    @property
    def has_aggregate_target(self) -> bool:
        return self.aggregate_target is not None

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
        if (
            self.has_aggregate_target
            and population == self.target_name
            and population not in set(self._data[self.population_col].unique())
        ):
            raise KeyError(
                f"Population '{population}' is an aggregate target and has no "
                "patient-level rows. Use matching_data.aggregate_target instead."
            )
        pop = self._data[self._data[self.population_col] == population]
        if not len(pop):
            raise KeyError(f"Population {population} not found!")
        return pop

    @property
    def populations(self) -> List[str]:
        """
        List of all populations present in the MatchingData object.
        """
        pops = set(self[self.population_col].unique().tolist())
        if self.has_aggregate_target:
            pops.add(self.target_name)
        return sorted(pops)

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
        headers = copy.deepcopy(self.headers)
        if self.has_aggregate_target:
            return MatchingData(
                pool=copy.deepcopy(self._data),
                target=copy.deepcopy(self.aggregate_target),
                headers=headers,
                population_col=self.population_col,
                pool_name=self.pool_name,
                target_name=self.target_name,
            )
        return MatchingData(
            data=copy.deepcopy(self.data),
            headers=headers,
            population_col=self.population_col,
            pool_name=self.pool_name,
            target_name=self.target_name,
        )

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

        if self.has_aggregate_target:
            target_col = pd.Series(index=agg.index, dtype=float)
            for feature, stat in agg.index:
                stats = self.aggregate_target.numeric.get(feature, {})
                if stat in stats:
                    target_col.loc[(feature, stat)] = stats[stat]
            agg[self.target_name] = target_col.round(decimals=2)

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
        pop_columns = list(counts.columns)

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

        out = pd.concat(out).fillna(0).astype(float)

        # normalize (or not) the patient-level population columns only; the
        # aggregate target column (if any) is populated separately below
        # since its rates/counts are already known exactly.
        if normalize:
            for c in pop_columns:
                n = counts.iloc[0][c]
                out[c] = out[c] / n
                out.loc[(f"{self.population_col} size", "N"), c] = n
        else:
            out = out.astype(int)

        if self.has_aggregate_target:
            out[self.target_name] = 0.0
            out.loc[(f"{self.population_col} size", "N"), self.target_name] = (
                self.aggregate_target.n
            )
            for cat in self.headers["categoric"]:
                rates = self.aggregate_target.categoric.get(cat, {})
                for value, rate in rates.items():
                    out.loc[(cat, value), self.target_name] = (
                        rate if normalize else round(rate * self.aggregate_target.n)
                    )
            # A feature the target does not disclose at all is unconstrained, not 0%.
            undisclosed = [
                cat
                for cat in self.headers["categoric"]
                if cat not in self.aggregate_target.categoric
            ]
            for cat in undisclosed:
                out.loc[cat, self.target_name] = float("nan")
            if not normalize:
                out[self.target_name] = out[self.target_name].astype(
                    "Int64" if undisclosed else int
                )

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
    if matching_data.has_aggregate_target:
        resolved_pool_name = pool_name or matching_data.pool_name
        resolved_target_name = target_name or matching_data.target_name
        if resolved_target_name == matching_data.target_name:
            raise ValueError(
                "Cannot split patient-level target rows from MatchingData with an "
                "aggregate target. Use matching_data.aggregate_target and "
                f"matching_data.get_population('{resolved_pool_name}') instead."
            )

    if isinstance(target_name, str) and isinstance(pool_name, str):
        target = matching_data.get_population(target_name)
        pool = matching_data.get_population(pool_name)
    elif isinstance(target_name, str) or isinstance(pool_name, str):
        if len(matching_data.populations) != 2:
            raise ValueError(f"""
            Cannot split into exactly two populations based on {matching_data.population_col}.
            Found populations: {','.join(matching_data.populations)}.
            """)
        if isinstance(target_name, str):
            pool_name = [p for p in matching_data.populations if p != target_name][0]
        if isinstance(pool_name, str):
            target_name = [p for p in matching_data.populations if p != pool_name][0]
        target = matching_data.get_population(target_name)
        pool = matching_data.get_population(pool_name)
    else:
        if len(matching_data.populations) != 2:
            raise ValueError(f"""
            Cannot split into exactly two populations based on {matching_data.population_col}.
            Found populations: {','.join(matching_data.populations)}.
            """)
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

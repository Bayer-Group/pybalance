Core Utilities
================

Matching Data
--------------------------------------
.. autoclass:: pybalance.utils.MatchingHeaders
    :members:

.. autoclass:: pybalance.utils.MatchingData
    :members:

.. autofunction:: pybalance.utils.infer_matching_headers

.. autofunction:: pybalance.utils.split_target_pool

Preprocessing
--------------------------------------
.. autoclass:: pybalance.utils.BaseMatchingPreprocessor
    :members: fit, _fit, _transform, _get_output_headers, _get_feature_names_out

.. autoclass:: pybalance.utils.CategoricOneHotEncoder
    :members:

.. autoclass:: pybalance.utils.NumericBinsEncoder
    :members:

.. autoclass:: pybalance.utils.DecisionTreeEncoder
    :members:

.. autoclass:: pybalance.utils.ChainPreprocessor
    :members:


Balance Calculators
--------------------------------------

.. autoclass:: pybalance.utils.BaseBalanceCalculator
    :members:

.. autoclass:: pybalance.utils.BetaBalance
    :members:

.. autoclass:: pybalance.utils.BetaSquaredBalance
    :members:

.. autoclass:: pybalance.utils.BetaMaxBalance
    :members:

.. autoclass:: pybalance.utils.GammaBalance
    :members:

.. autoclass:: pybalance.utils.GammaSquaredBalance
    :members:

.. autoclass:: pybalance.utils.GammaXTreeBalance
    :members:

.. autofunction:: pybalance.utils.BalanceCalculator

.. autoclass:: pybalance.utils.BatchedBalanceCalculator
    :members:

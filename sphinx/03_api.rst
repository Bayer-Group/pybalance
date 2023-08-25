*****************
API documentation
*****************

.. contents:: Table of Contents


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

.. autoclass:: pybalance.utils.BatchedBalanceCaclulator
    :members:


Matchers
================

Propensity Score Matcher
--------------------------------------
.. autoclass:: pybalance.propensity.PropensityScoreMatcher
    :members: match

.. autofunction:: pybalance.propensity.plot_propensity_score_match_distributions

.. autofunction:: pybalance.propensity.plot_propensity_score_match_pairs

Genetic Matcher
--------------------------------------
.. autoclass:: pybalance.genetic.GeneticMatcher
    :members:

.. autofunction:: pybalance.genetic.get_global_defaults

Constraint Satisfaction Matcher
--------------------------------------
.. autoclass:: pybalance.lp.ConstraintSatisfactionMatcher
    :members:


Visualization
================

.. autofunction:: pybalance.visualization.plot_numeric_features

.. autofunction:: pybalance.visualization.plot_categoric_features

.. autofunction:: pybalance.visualization.plot_binary_features

.. autofunction:: pybalance.visualization.plot_joint_numeric_distributions

.. autofunction:: pybalance.visualization.plot_joint_numeric_categoric_distributions

.. autofunction:: pybalance.visualization.plot_per_feature_loss


Simulation
================
.. autofunction:: pybalance.sim.generate_toy_dataset

.. autofunction:: pybalance.sim.load_paper_dataset

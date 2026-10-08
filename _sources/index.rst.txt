**************************
PyBalance Utilities
**************************

`pybalance` is a python library for confounding adjustment in non-randomized
populations. Given a "pool" and a "target" population that differ
systematically on a set of covariates, `pybalance` finds either a **matched
subset** of the pool or a set of **weights** on the pool that make it resemble
the target -- a key step in any causal inference analysis built on
observational data.

Everything in `pybalance` is driven by a researcher-defined balance metric
(e.g. standardized mean difference), which is optimized directly rather than
treated as a side effect of some other model. This includes propensity score
matching: instead of fitting one model and hoping it balances well,
`pybalance` searches over propensity-model hyperparameters and keeps whichever
fit optimizes the chosen balance metric. See the :doc:`00_introduction` for
the underlying problem formulation.

Features
========

- Matching: linear program (integer solver), evolutionary/genetic search, and
  propensity score matching, plus matching against an aggregate (published
  Table 1) target.
- Weighting: Matching-Adjusted Indirect Comparison (MAIC), general entropy
  balancing, and propensity-score (IPTW) weighting.
- A variety of balance calculators for defining and measuring balance.
- Visualization tools for inspecting covariate balance before/after.
- Dataset simulation utilities for testing and demos.

Get started by following the :doc:`01_installation` instructions, then work
through the :doc:`02_demos`. Questions or issues are welcome on
`GitHub <https://github.com/Bayer-Group/pybalance>`_.


..	toctree::
	:maxdepth: 3
	:caption: Getting Started

	00_introduction
	01_installation

..	toctree::
   :maxdepth: 2
   :caption: Demos - Core

   demos/core_01_matching_data.ipynb
   demos/core_02_balance_calculators.ipynb

..	toctree::
   :maxdepth: 2
   :caption: Demos - Matching

   demos/matching_01_propensity.ipynb
   demos/matching_02_linear_program.ipynb
   demos/matching_03_cardinality.ipynb
   demos/matching_04_genetic.ipynb
   demos/matching_05_aggregate.ipynb

..	toctree::
   :maxdepth: 2
   :caption: Demos - Weighting

   demos/weighting_01_iptw.ipynb
   demos/weighting_02_maic.ipynb

..	toctree::
    :maxdepth: 2
    :caption: API

    03_api

..	toctree::
	:maxdepth: 2
	:caption: License & Help

	04_license
	05_help


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

# PyBalance

The `pybalance` library implements several routines for optimizing the balance
between non-random populations. In observational studies, this matching process
is a key step towards minimizing the potential effects of confounding
covariates. The official documentation is hosted [here](https://bayer-group.github.io/pybalance/).
An application of this library to matchng in the pharmaceutical setting is presented here: [here](https://onlinelibrary.wiley.com/doi/10.1002/pst.2352).

## Features

- Implements linear and non-linear optimization approaches for matching.
- Utilizes integer program solvers and evolutionary solvers for optimization.
- Includes implementation of propensity score matching for comparison.
- Offers a variety of balance calculators and matchers.
- Provides visualization tools for analysis.
- Supports simulation of datasets for testing and demonstration purposes.

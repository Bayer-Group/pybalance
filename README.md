![https://bayer-group.github.io/pybalance/index.html](logo.png)

## Confounding Adjustment

In scientific experiments, researchers aim to identify cause and effect by
holding all variables except one constant. Any difference in outcome can then
be attributed to the manipulated variable.

However, in many practical cases, it is not possible to control the variables
of interest. For instance, it is unethical to conduct a randomized trial to test
the effects of smoking on long-term health outcomes; yet the answer to this
question is of extreme interest to policy makers, insurance companies and
regulatory agencies. Similarly, in social science research, when studying the
impact of education on income, researchers cannot manipulate individuals' education
levels while holding all other variables constant.

In these cases, observational data can form the basis for "natural experiments" but
care must be taken in interpreting these data. One major issue with interpreting these
data is known as "confounding".

A classic example of confounding is the association between coffee consumption and
heart disease. Initially, a study might find a positive correlation between high
coffee consumption and increased risk of heart disease. However, this apparent
relationship could be confounded by the fact that heavy coffee drinkers are more
likely to also smoke, which is a known risk factor for heart disease. In this case,
smoking acts as a confounding variable, as it distorts the true relationship between
coffee consumption and heart disease. To address this, researchers need to adjust for
smoking status and potentially other relevant variables to accurately assess the
independent impact of coffee consumption on heart disease risk.

In general, any comparative analysis of two non-randomized populations will differ
systematically in a number of covariate dimensions and these systematic differences
must be adjusted for as part of any causal inference analysis. That is where
`pybalance` comes in.

## PyBalance

`pybalance` is a suite of tools in python for performing confounding adjustment
in non-randomized populations. In `pybalance`, we start with measures of "balance"
(how similar two populations are) and directly optimize this metric. This approach is
different, and we think almost always better, from the well-known propensity score
approach, in which the probability of treatment assignment is modelled, but balance metrics
are almost always anyway implicitly defining the success criterion
(see our [demo](https://bayer-group.github.io/pybalance/demos/ps_matcher.html)).
Our approach here is to explicitly define and directly optimize the balance metric that
is relevant for the given problem.

The `pybalance` library implements several routines for optimizing balance. To learn more
about these methods, head on over to the
[demos](https://bayer-group.github.io/pybalance/02_demos.html). Then give
the code a spin for yourself by following the
[installation instructions](https://bayer-group.github.io/pybalance/01_installation.html).
Any questions or issues please feel free to open an [issue](https://github.com/Bayer-Group/pybalance/issues)
or start a [discussion](https://github.com/Bayer-Group/pybalance/discussions).

An application of this library to build an external control arm in a pharmaceutical
setting is presented [here](https://onlinelibrary.wiley.com/doi/10.1002/pst.2352).

## Features

- Implements linear and non-linear optimization approaches for matching.
- Utilizes integer program solvers and evolutionary solvers for optimization.
- Includes implementation of propensity score matching for comparison.
- Offers a variety of balance calculators and matchers.
- Provides visualization tools for analysis.
- Supports simulation of datasets for testing and demonstration purposes.

## Running Streamlit Demos

1. Install the pip packages using `pip install -r environments/requirements.txt`.
2. Run the FastAPI server using `python bin/api.py`.
3. Run the Streamlit server using `streamlit run bin/streamlit.py`.

## Limitations

At the moment, `pybalance` only implements matching routines. Suport for weighting
methods is on our roadmap and will appear in a future release.

## Citation

If you use `pybalance` for your research, please acknowledge this with a citation to
our [paper](https://onlinelibrary.wiley.com/doi/10.1002/pst.2352).

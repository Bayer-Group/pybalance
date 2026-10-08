Introduction
^^^^^^^^^^^^
The `pybalance` library implements several routines for optimizing the
balance between non-random populations. In observational studies, this
process is a key step towards minimizing the potential effects of confounding
covariates, and is a core tool for causal inference whenever randomization
isn't possible.

`pybalance` supports two ways of achieving balance: **matching**, which draws
a subset of the pool that resembles the target, and **weighting**, which
instead reweights every pool subject so that the weighted pool resembles the
target. Both approaches work by balancing covariate distributions directly,
without specifying to whom a given individual is matched.


Problem Statement
=================

Consider two groups of study subjects, together with a set of :math:`F`
covariates (e.g., age, height, smoker/non-smoker) describing the characteristics
of the two groups. By convention, we refer to the smaller group as the "target"
population and the larger group as the "pool". Our goal is to draw subjects from
the pool such that the chosen subset "matches" (to be defined) as close as
possible the target population.

More formally, given a pool of size :math:`M` and a target of size :math:`N`,
our goal is to choose :math:`N` patients from the pool that best resemble the
target population. Since there are :math:`M \choose N` such possible subsets,
exploring the whole space of solutions is generally infeasible for even
modestly-sized matching problems.

Depending on the nature of the matching measure, different solvers are
available. For instance, in the case of minimizing the mean standardized error
between the covariates, one can formulate the optimization as follows.

**Define**:

..	math::
	:label: eq1
	:nowrap:

	\begin{equation}
	x_{m} =
	\begin{cases}
	1, & \mbox{if patient m}\mbox{ is selected} \\
	0, & \mbox{otherwise.}
	\end{cases}
	\end{equation}

Take :math:`c_{mf}` to be the value of feature :math:`f` corresponding to patient's :math:`m`.

..	math::
	:label: cost1
	:nowrap:

	\begin{align*}
		a_f = \bigg| \sum_{m=1}^{M} x_{m}c_{mf} - \sum_{m=1}^N c_{mf} \bigg|,&\\
		Minimize~\sum_{f=1}^{F}a_f:& \\
		\mbox{Subject to :}\sum_{m=1}^{M} x_{m} = N.
	\end{align*}

In this case, since the objective function and constraints are linear, fast
integer program solvers can be used as a backend to solve for the best matching
population. The cost function defined in :eq:`cost1` is solved using SAT solver
library from `Google Or-Tools Sat
<https://developers.google.com/optimization/cp/cp_solver>`_.

Note that in this formulation, we do not explicitly assign a control patient to
a treatment patient, therefore, the decision variable :math:`x` is simply a
vector with a length equal to the number of patients in the control group. This
detail allows the integer program solver to scale to relatively large matching
problems.

However, linearity in the objective function is not always desireable, since
improving a poorly matched dimension slightly is often better than improving a
well-matched dimension by the same amount. Non-linear objective functions can
enforce this prior on the solution space, but require different optimization
methods. In this case, we can apply an evolutionary solver, which stochastically
searches the solution space. An evolutionary solver is also implemented in
`pybalance`, together with a number of heuristics for efficiently
searching the space.

For completeness and ease of comparison, `pybalance` also implements matching
based on propensity score. For greater technical detail as well as applications,
see our publication `here
<https://onlinelibrary.wiley.com/doi/10.1002/pst.2352>`_.


Weighting: Matching With Real-Valued Weights
=============================================

Matching is a special case of a more general problem: instead of restricting
each pool subject to be either fully included (:math:`x_m=1`) or fully
excluded (:math:`x_m=0`), we can let every subject keep a non-negative,
real-valued weight :math:`w_m \geq 0`. The balance constraint from
:eq:`cost1` becomes

..	math::
	:label: cost2
	:nowrap:

	\begin{align*}
		a_f = \bigg| \sum_{m=1}^{M} w_{m}c_{mf} - \sum_{m=1}^N c_{mf} \bigg|,&\\
		Minimize~\sum_{f=1}^{F}a_f:& \\
		\mbox{Subject to :}\sum_{m=1}^{M} w_{m} = N,\ w_m \geq 0.
	\end{align*}

Put side by side with :eq:`cost1`, the only thing that has changed is the
domain of the decision variable: :math:`x_m \in \{0, 1\}` for matching versus
:math:`w_m \in [0, \infty)` for weighting. In other words, matching is just
weighting with integer (0/1) weights -- a nice way to see why both live in the
same library and share the same balance calculators.

Letting weights move continuously rather than snapping to 0 or 1 relaxes the
combinatorial search into a smooth optimization problem, which is solved very
differently in practice (e.g. entropy balancing, propensity-score/IPTW
weights) but pursues the same goal: making the weighted pool resemble the
target as closely as possible. The practical trade-off is that matching
discards non-matched subjects entirely (simple to reason about, but throws
away data and reduces precision), while weighting keeps everyone but can
produce large weights for poorly-overlapping subjects (uses all the data, at
the cost of a sometimes-fragile effective sample size). See the
`Weighting demos <02_demos.html>`_ for the methods `pybalance` implements
(MAIC, entropy balancing, IPTW).

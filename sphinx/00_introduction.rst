Introduction
^^^^^^^^^^^^
The `pybalance` library implements several routines for optimizing the
balance between non-random populations. In observational studies, this matching
process is a key step towards minimizing the potential effects of confounding
covariates.

The library implements two separate approaches to matching, both of which work
by balancing covariate distributions directly, without specifying to whom a
given individual is matched.


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

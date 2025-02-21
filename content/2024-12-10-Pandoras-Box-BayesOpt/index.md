+++
title = "Cost-aware Bayesian Optimization via the Pandora's Box Gittins Index"
[extra]
authors = [
    {name = "Qian Xie", url = "https://qianjanexie.github.io"},
    {name = "Raul Astudillo", url = "https://raulastudillo.netlify.app"},
    {name = "Peter I. Frazier", url = "https://people.orie.cornell.edu/pfrazier/"},
    {name = "Ziv Scully", url = "https://ziv.codes"},
    {name = "Alexander Terenin", url = "https://avt.im/"},
]
venue = {name = "NeurIPS", date = 2024-12-10, url = "https://neurips.cc/Conferences/2024"}
buttons = [
    {name = "Paper", url = "https://papers.nips.cc/paper_files/paper/2024/hash/d14c355d5e88cff437a6303d2d716252-Abstract-Conference.html"},
    {name = "PDF", url = "https://arxiv.org/pdf/2406.20062"},
    {name = "Code", url = "https://github.com/QianJaneXie/PandoraBayesOpt"},
]
katex = true
large_card = true
+++

**TLDR**: We develop a new algorithm for Bayesian optimization based on the *Gittins index*, a concept from economics, queueing, and Markovian bandits. 
It naturally handles heterogeneous costs, and we show it has strong empirical performance compared to a comprehensive set of baselines.

---

Bayesian optimization is everywhere: from hyperparameter tuning,[^hyperparameter-tuning] to AI for science applications,[^ai-for-science] to systems like AlphaGo[^alphago] that have captured the imagination of the general public, Bayesian optimization is well-recognized as an effective way to perform black-box global optimization in settings where efficiency is key.
At the core of Bayesian optimization algorithms is the question of *how one should design the acquisition function*, particularly in realistic settings which incorporate evaluation costs and other factors that go beyond the classical framework.

In this work, we introduce a novel cost-aware acquisition function design framework.
To do so, we connect a certain variant of cost-aware Bayesian optimization with the *Pandora's Box* problem from economics—a decision problem whose solution can be reinterpreted as an acquisition function.
Using this connection, we propose the *Pandora's Box Gittins Index* acquisition function for general cost-aware Bayesian optimization problems, which we name so due to its connection with Gittins Index Theory, an abstract framework for deriving acquisition-function-like decision rules for certain classes of decision problems.
We show this acquisition function performs strongly, especially on problems of moderate-to-high dimension.
Our work takes a first step towards bringing ideas from Gittins Index Theory into Bayesian optimization, which we optimistically believe can have significant scope as an acquisition function design framework for cost-aware problems.


# Cost-aware Bayesian Optimization

In Bayesian optimization,[^bayesopt-book] [^bayesopt-survey] we are interested black-box global optimization of an unknown function `$f:X\to\mathbb{R}$`, which we model using a Gaussian process.[^gp-book]
One can formulate a number of *cost-aware* variants of Bayesian optimization,[^cost-aware] which additionally incorporate a cost function `$c:X\to\mathbb{R}_+$` which models the cost of obtaining another sample.
For instance, in the *expected budget-constrained* variant of the problem, we are interested in algorithms which achieve a small expected *simple regret*
```
$$
\mathbb{E} \sup_{x\in X} f(x) - \mathbb{E} \sup_{1\leq t\leq T} f(x_t)
$$
```
subject to the budget constraint `$\mathbb{E} \sum_{t=1}^T c(x_t) \leq B$`, which holds in expectation.
We allow the algorithm to decide when to stop sampling: we denote the stopping time by `$T$`, and once the algorithm stops it returns the best point observed so far.

Our starting point is to ask: *are there simplified settings where one can analytically derive the optimal algorithm—that is, the one that achieves the smallest expected simple regret*?
As an example, in the non-cost-aware setting, one can derive the classical *expected improvement* acquisition function by considering a one-step greedy approximation to a dynamic program defined using simple regret.[^expected-improvement]
In our paper, we show there is a second, different simplification that leads to an analytic solution—of a spatial rather than temporal character—which we now describe.


## Pandora's Box

The Pandora's Box problem is a decision-making problem which originally arose in economics.[^pandoras-box]
Suppose that a decision-making agent is presented with a collection of closed boxes, labeled `$X=\{1,..,N\}$`. 
Each box has a reward `$f(x)$` inside, with a known distribution—for instance, Gaussian with mean `$\mu(x)$` and variance `$\sigma^2(x)$`.
The rewards inside all boxes are independent.
The agent is allowed to pay a cost `$c(x)$` to open any of the closed boxes, at which point the reward inside the box is revealed.
The agent is also allowed to take a reward from at most one open box, which ends the decision-problem.
The agent's total value is therefore
```
$$
\mathbb{E} \sup_{1\leq t\leq T} f(x_t) - \sum_{t=1}^T c(x_t)
$$
```
where `$T+1$` is the time at which the agent decided to take a reward from an open box.[^best-open-box]
Figure 1 illustrates the setup.


{% figure(alt=["Pandora's Box"] src=["pandoras_box.svg"] dark_invert=[true]) %}
**Figure 1.** A visual illustration of the Pandora's Box problem, with two closed boxes and one open box.
{% end %}

This already looks a lot like expected budget-constrained optimization, but with two changes:

1. In Bayesian optimization, the set `$X$` need not be finite, and the objective function values `$f(x)$` and `$f(x')$` for `$x \neq x'$` can be correlated.
2. Instead of incorporating costs using an expected budget constraint, we add them to the simple regret objective.

Of these differences, the first is significant: correlations are what allow us to perform Bayesian optimization with Gaussian processes which model smooth functions.
Given the importance of smoothness, this difference therefore indicates we've departed some distance from the Bayesian optimization setup we started with.
So, why should one even consider what happens without correlations?
One reason, stated in our language and notation, is as follows.

**Theorem (Weitzman).**[^pandoras-box] 
The optimal policy of the Pandora's Box problem takes the form of maximizing the acquisition function `$\alpha^\star$`, defined as
```
$$
\alpha^\star(x) = g \quad\text{where}\ g\ \text{solves}\quad \operatorname{EI}_f(x;g) = c(x)
$$
```
where `$\operatorname{EI}_\psi(x;y) = \mathbb{E} \max(0, \psi(x) - y)$` is the expected improvement function.

This means that the optimal policy in the Pandora's Box problem *takes the form of maximizing an acquisition function*.
This specific acquisition function `$\alpha^\star(x)$` is defined in terms of a root-finding problem whose objective resembles the classical expected improvement acquisition function.

Before proceeding, let us also address the difference between the expected budget-constrained setup we started with, and the cost-per-sample setup used in the Pandora's Box problem.
In short, if tie-breaking is handled correctly, both problems have the same solution as consequence of Lagrangian duality: we show one can solve the cost-per-sample Pandora's Box with costs `$\lambda c(\cdot)$`, where `$\lambda$` is the Lagrange multiplier, and obtain an optimal policy for an expected-budget-constrained Pandora's Box.[^lagrangian-duality]
This means correlations are the essential difference between the two setups.


## Solving Pandora's Box

Why does the solution to Pandora's Box take the form of maximizing an acquisition function?
What is the meaning of the root-finding problem which defines `$\alpha^\star$`?
One way to approach these questions is to first consider a simpler problem, involving just two boxes: one closed box with reward `$f$` and cost `$c$`, and one open box with reward `$g$`. Figure 2 illustrates this.

{% figure(alt=["Pandora's Box"] src=["pandoras_box_comparison.svg"] dark_invert=[true]) %}
**Figure 2.** A simplified Pandora's Box problem, with one closed and one open box.
{% end %}

Now, there's only one decision: we can either open the box, or leave it closed.
Let's see what the value is in both cases:

1. If we open the box, our reward is `$\mathbb{E} \max(f, g) - c$`.
2. If we don't open the box, our reward is `$g$`.

In the former expression, the maximum appears because we can only take one reward, and always choose to take the larger one.
From these expressions, it is easy to see that it is optimal to open the box if `$\mathbb{E}\max(f-g,0) \geq c$`. 
In other words, it is optimal to open the closed box if its expected improvement is larger than its cost to open.

Now, we ask: *what would `$g$` need to be in order for both actions to be optimal?*
In such a situation, one can think of the closed box and open box as equivalent to one another, since the same expected reward is obtained no matter what decision is made.
As consequence, the value of `$g$` for which `$\mathbb{E}\max(f-g,0) = c$` can be thought of as a kind of *fair price* or *fair value* for the closed box, depending on the sign convention in use.

The insight of Weitzman[^pandoras-box]—and indeed, of Gittins,[^gittins-index] who discovered the same idea in a much more general setting—is that the original Pandora's Box problem can be solved by replacing closed boxes with equivalent open boxes one-by-one without affecting the optimal decision.
Using this procedure, the optimal decision is to order all boxes by their fair value, and open boxes according to this order until all remaining fair values are smaller than the best observed reward.

In addition to the argument sketched here, one can show the same result using different techniques, for instance via analytic arguments based on surrogate prices.[^analytic-argument]
There is both a rich literature on the Pandora's Box problem, studying these[^analytic-argument-paper] and related questions.[^pandoras-box-optional-inspection]
This result we consider is an instance of a Gittins Index Theorem[^gittins-index-book]—a broad class of results which can be thought of as ways to assign fair values to stochastic objects, and in addition to economics are also a central part of queueing theory.[^gittins-queueing]
For a clean introduction to these ideas, we recommend starting from the so-called *golf paper*,[^gittins-intro-golf-paper] which describe a more general form of the one-closed-box and one-open-box viewpoint described here.


## The Pandora's Box Gittins Index Acquisition Function

Our paper's primary contribution is to bring these ideas to Bayesian optimization.
For this, we need to handle correlations, which we propose to do in the obvious way: by plugging in the correlated posterior distribution into `$\alpha^\star$`.
This gives the *Pandora's Box Gittins Index* acquisition function 
```
$$
\alpha^{\operatorname{PBGI}}(x) = g \quad\text{where}\ g\ \text{solves}\quad \operatorname{EI}_{f\mid y}(x;g) = c(x)
$$
```
defined as the solution of a root-finding problem whose objective involves the posterior distribution and cost function.
One can use this in both cost-aware and classical settings, in the latter case by having the costs be a constant function whose value is treated as a hyperparameter.
Our paper describes a number of properties of this acquisition function, such as its gradient, and connections with other acquisition functions such as UCB and expected improvement.
Let's see how this acquisition function performs.


# Experiments

Below, we investigate the behavior of `$\alpha^{\operatorname{PBGI}}$`.
This acquisition function depends on the posterior through its mean and standard deviation: we can therefore examine how it behaves for different values of `$c(x) = \lambda$` in the uniform-cost case.
Figure 3 shows this: we see that, roughly, `$\lambda$` controls risk-seeking vs. risk-averse behavior, as indicated by large-`$\lambda$` variants initially achieving smaller regret, which is later overtaken by small-`$\lambda$` variants, in terms of medians averaged over 256 seeds.
We also see in the `$\lambda\to0$` limit that the contour plot becomes linear, indicating that `$\alpha^{\operatorname{PBGI}}$` begins to behave like the UCB acquisition function in this limit.
Our paper contains more discussion on this connection, as well as an additional dynamic-`$\lambda$` variant.

{% figure(alt=["Contour and Cost"] src=["contour_and_cost.svg"] dark_invert=[true]) %}
**Figure 3.** Illustration of how `$\alpha^{\operatorname{PBGI}}$` behaves when one varies the posterior mean and standard deviation, as well as costs.
{% end %}

Next, we benchmark the Pandora's Box Gittins Index acquisition function on a comprehensive set of experiments to evaluate its performance relative to baselines, under both varying costs, and a more-classical setting where the cost function is a constant and is treated as a hyperparameter.
Full experimental details are in the paper, along with descriptions of baselines and results on sensitivity to various problem and setup hyperparameters.
We deliberately use the same tuning for our algorithm in every problem to ensure performance differences are not due primarily to tuning, and we plot medians and quartiles with respect to 16 random seeds in each experiment to indicate variability.
Figure 4 shows performance on optimizing random functions sampled from the prior, where we use a squared exponential prior with length scale is `$\kappa = 10^{-1}$`.
Our paper includes similar plots for a set of synthetic benchmark functions and empirical optimization objectives.

We see from Figure 4 that behavior varies according to three regimes: on sufficiently-easy low-dimensional problems, most algorithms perform similarly relative to variability.
As one increases dimension, a regime where `$\alpha^{\operatorname{PBGI}}$` decisively outperforms most baselines starts to emerge, with exception of UCB in the uniform-cost version of this experiment. 
In our paper, we see similar behavior on other problem classes—except that the second-best-performing baselines does not always perform comparably, and is different in different experiments.
Eventually, however, the dimension becomes large enough that no method performs better than random search.

{% figure(alt=["Bayesian Regret"] src=["bayesian_regret.svg"] dark_invert=[true]) %}
**Figure 4.** Performance in terms of Bayesian regret: that is, with respect to optimizing objective functions randomly sampled from the prior.
{% end %}


# Future Work

Beyond vanilla cost-aware Bayesian optimization, we think approaches like ours can give rise to a broad class of acquisition functions for cost-aware settings with more complicated forms of feedback.
For instance, one can consider analogues of the one-closed-box and one-open-box setup, and replace the closed box with a general stochastic process.
In such settings, Gittins Index Theory continues to apply,[^gittins-intro-golf-paper] and the optimal policy for the discrete, uncorrelated problem takes the form of an acquisition function. 
Just as for Pandora's Box, we can consider applying this acquisition function more generally, to the original correlated cost-aware setting.
The challenge becomes how to compute the acquisition function, since we no longer expect the root-finding-problem to admit an analytic expected-improvement-type objective.
We think this approach may be a promising angle of attack for finding acquisition functions for various more complex forms of cost-aware Bayesian optimization, such as freeze-thaw[^freeze-thaw] and other queueing-flavor setups, as well as multi-fidelity[^multi-fidelity] and other advanced settings.


# Conclusions

In this work, we connected cost-aware Bayesian optimization with the Pandora's Box problem from economics.
We used this connection to propose the Pandora's Box Gittins Index—an acquisition function for cost-aware Bayesian optimization.
Our experiments showed that the Pandora's Box Gittins Index performs well on problems that are sufficiently difficult that performance differences from various baselines are visible, but not so difficult that no method can make meaningful progress.
Our acquisition function is an instance of a Gittins index—a broad framework for assigning fair values or fair prices to stochastic objects, which we think can also be applied, in future work, to other cost-aware Bayesian optimization setups beyond our setup.

# References

[^hyperparameter-tuning]: J. Snoek, H. Larochelle, and R. P. Adams. Practical Bayesian Optimization of Machine Learning Algorithms. *Advances in Neural Information Processing Systems*, 2012.

[^ai-for-science]: See for instance [these](https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/global-stochastic-optimization-of-stellarator-coil-configurations/7377A84A021E2643DD7EB9CB2F0D6F1E) [two](https://journals.aps.org/prab/abstract/10.1103/PhysRevAccelBeams.27.084801) papers, which apply Bayesian optimization to various problems arising in nuclear fusion research and particle accelerator settings.

[^alphago]: See [this tweet](https://twitter.com/NandoDF/status/1791204574004498729) by Nando de Freitas, which calls Bayesian optimization one of the "secret ingredients" of AlphaGo—one which improved its win rate from 50% to 66.5% in self-play games through better hyperparameter tuning.

[^bayesopt-book]: R. Garnett. Bayesian Optimization. Cambridge University Press, 2023.

[^bayesopt-survey]: P. I. Frazier. Bayesian Optimization. *Recent Advances in Optimization and Modeling of Contemporary Problems*, 2018.

[^gp-book]: C. E. Rasmussen and C. K. I. Williams. Gaussian Processes for Machine Learning. MIT Press, 2006.

[^cost-aware]: E. H. Lee, D. Erksson, V. Perrone, and M. Seeger. A Nonmyopic Approach to Cost-
constrained Bayesian Optimization. *Uncertainty in Artificial Intelligence*, 2020.

[^expected-improvement]: See Chapter 7 of the *Bayesian Optimization* book.

[^pandoras-box]: M. L. Weitzman. Optimal Search for the Best Alternative. *Econometrica*, 1979.

[^best-open-box]: If there is more than one open box, we assume the agent always takes the largest reward among all open boxes, since this leads to a higher value and is therefore the optimal decision in this specific situation. We also need to handle tie-breaking: see the paper's appendix for details on this.

[^lagrangian-duality]: Our results extend those in [this paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4347447), whose authors prove the same result in a certain extension Pandora's Box under the assumption of discrete support. We prove the result for only the Pandora's Box model, but our argument allows for continuous support—a detail which makes the argument significantly more challenging due to subtleties involving envelope theorems. A discussion on this can be found in the paper.

[^gittins-index]: J. C. Gittins. Bandit Processes and Dynamic Allocation Indices. *Journal of the Royal Statistical Society, Series B: Statistical Methodology*, 1979.

[^analytic-argument]: A sketch of this argument for Pandora's Box is given in [this blog post](https://www.bowaggoner.com/blog/2018/07-20-pandoras-box/) by Bo Waggoner.

[^analytic-argument-paper]: R. Kleinberg, B. Waggoner, and E. G. Weyl. Descending Price Optimally Coordinates Search.
*Economics and Computation*, 2016.

[^pandoras-box-optional-inspection]: L. Doval. Whether or Not to Open Pandora's Box. *Journal of Economic Theory*, 2018.

[^gittins-index-book]: J. C. Gittins, K. D. Glazebrook, and R. R. Weber. Multi-armed Bandit Allocation Indices. Wiley, 2011.

[^gittins-queueing]: Z. Scully, I. Grosof, and M. Harchol-Balter. The Gittins Policy is Nearly Optimal in the
M/G/k under Extremely General Conditions. *Measurement and Analysis of Computing Systems*, 2020.

[^gittins-intro-golf-paper]: I. Dumitriu, P. Tetali, and P. Winkler. On Playing Golf with Two Balls. *SIAM Journal on Discrete Mathematics*, 2003.

[^freeze-thaw]: *Freeze-thaw* refers to the setting where evaluating the unknown function results in one launching a computational job of some kind, which provides early feedback about what the final function value will be before the computation finishes. In this setting, computation can be paused or stopped to save time, if early results indicate that a good objective value is unlikely. See [this paper](https://arxiv.org/abs/1406.3896) for more information.

[^multi-fidelity]: See Chapter 11 of the *Bayesian Optimization* book.
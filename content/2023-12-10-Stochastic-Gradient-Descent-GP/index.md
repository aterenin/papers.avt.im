+++
title = "Sampling from Gaussian Process Posteriors using Stochastic Gradient Descent"
[extra]
authors = [
    {name = "Jihao Andreas Lin", url = "https://jandylin.github.io", star = true},
    {name = "Javier Antorán", url = "https://javierantoran.github.io", star = true},
    {name = "Shreyas Padhy", url = "https://shreyaspadhy.github.io", star = true},
    {name = "David Janz", url = "https://djanz.org"},
    {name = "José Miguel Hernández-Lobato", url = "https://jmhl.org"},
    {name = "Alexander Terenin", url = "https://avt.im/"},
]
venue = {name = "NeurIPS", date = 2023-12-10, url = "https://neurips.cc/Conferences/2023"}
buttons = [
    {name = "Paper", url = "https://arxiv.org/abs/2306.11589"},
    {name = "PDF", url = "https://arxiv.org/pdf/2306.11589"},
    {name = "Code", url = "https://github.com/cambridge-mlg/sgd-gp"},
]
katex = true
large_card = true
+++

Gaussian processes are a fundamental building block for decision-making systems such as Bayesian optimization.
Traditionally, their performance is limited by the need to solve large linear systems, which has cubic cost with respect to dataset size.
In this work, we explore using stochastic gradient algorithms for solving these linear systems: somewhat counterintuitively we find these algorithms produce strong empirical results *even in cases where they fail to converge*.
Let's try to understand why.

# Large-scale Gaussian processes and posterior asymptotics

To begin, let's first understand how the data distribution affects standard large-scale Gaussian process techniques.
In one dimension, we focus on two key data distribution regimes: *infill asymptotics* and *large-domain asymptotics*.
Infill asymptotics involve data sampled from a Gaussian which becomes more dense as size increases.
In contrast, *large-domain asymptotics* involve time-series data on a regular grid which takes up a larger region as size increases.
The key difference between these is that the former approximately fixed the volume of space which data takes up, and increases data-density as data size increases, whereas the latter fixes data-density and increases the volume of space the data takes up as data size increases.
Let's see a simple comparison between standard large-scale Gaussian process approximations in both regimes.

{% figure(alt=["Large-scale Gaussian process comparison"] src=["toy_comparison.svg"] dark_invert=[true]) %}
**Figure 1.** A comparison of standard large-scale Gaussian process techniques, including conjugate gradient and sparse Gaussian processes with variational inference, under both asymptotics.
{% end %}

From this comparison, one can see that different large-scale Gaussian process approximations work well in different regimes.
Conjugate-gradient-based Gaussian processes[^cg] work well under large-domain asymptotics, whereas sparse Gaussian processes trained via variational inference[^ip-s][^ip-v] work well under infill asymptotics.
One can show theory which suggests this this distinction holds beyond one-dimensional problems.[^ip-theory]
In contrast, the stochastic gradient descent variant we present looks very reasonable in both cases: it empirically converges in most regions of state space under infill asymptotics, and converges everywhere under large-domain asymptotics.
Let's look at this algorithm in more details.


# Stochastic gradient descent for posterior sampling

To formulate stochastic gradient descent for posterior sampling, let's begin by writing down a random quadratic optimization problem for computing posterior samples.
Let `$f \sim\mathrm{GP}(0,k)$` be the prior, and let `$\boldsymbol{y}\mid f\sim\mathrm{N}(f(\boldsymbol{x}), \mathbf\Sigma)$` be the likelihood.
Let's begin with the *pathwise conditioning*[^efficient-sampling][^pathwise-conditioning] formula for posterior random functions, namely

```
$$
(f\mid\boldsymbol{y})(\cdot) = f(\cdot) + \mu_{f\mid\boldsymbol{y}}(\cdot) - \mathbf{K}_{(\cdot)\boldsymbol{x}} (\mathbf{K}_{\boldsymbol{x}\boldsymbol{x}} + \mathbf\Sigma)^{-1}(f(\boldsymbol{x}) + \boldsymbol\varepsilon)
$$
```

where `$\mu_{f\mid\boldsymbol{y}}$` is the posterior mean, `$f \sim \mathrm{GP}(0,k)$` is a sample from the prior, and `$\varepsilon\sim\mathrm{N}(\boldsymbol{0},\mathbf\Sigma)$`.
Let's focus attention on computing posterior samples: the posterior mean will be handled similarly.
Consider the term `$\boldsymbol\alpha^* = (\mathbf{K}_{\boldsymbol{x}\boldsymbol{x}} + \mathbf\Sigma)^{-1}(f(\boldsymbol{x}) + \boldsymbol\varepsilon)$`, which we call the *representer weights* for sampling.
Assume for simplicity that `$\mathbf\Sigma = \sigma^2\mathbf{I}$`.
Since `$\boldsymbol\alpha^*$` solves a random linear system, one can write it as 

```
$$
\boldsymbol\alpha^* = \operatorname*{\arg\min}_{\boldsymbol\alpha \in \mathbb{R}^N} \frac{1}{\sigma^2} \sum_{i=1}^N (f(x_i) + \varepsilon_i - \mathbf{K}_{x_i \boldsymbol{x}} \boldsymbol\alpha)^2 + \Vert\boldsymbol\alpha\Vert_{\mathbf{K}_{\boldsymbol{x}\boldsymbol{x}}}^2
$$
```

We can stochastically estimate the large sum using minibatches.
Similarly, we can apply a Fourier-feature-based stochastic estimator for the squared norm term.
We use *efficient sampling* to approximately sample the prior `$f(x_i)$` using Fourier features.[^efficient-sampling][^pathwise-conditioning]
This gives us a subquadratic stochastic estimator for this optimization objective which is almost unbiased, in the sense that the only bias present is from efficiently sampling the prior.
To reduce this objective's variance, we apply a number of tricks, including carefully shifting the `$\boldsymbol\varepsilon$` noise term into the regularizer, which are described in the paper.
The result is a practical stochastic optimization objective for Gaussian process posterior samples.


## SGD doesn't converge in practice, but still performs well

With the optimization objective and stochastic estimators defined, all that remains is to specify the precise algorithm.
In the paper, use stochastic gradient descent with Nesterov momentum, gradient clipping, arithmetic Polyak averaging, and a constant learning rate.
Let's see how this algorithm performs, in particular how it is affected by observation noise in the likelihood.

{% figure(alt=["Convergence of stochastic gradient descent for the Gaussian process mean"] src=["exact_metrics.svg"] dark_invert=[true]) %}
**Figure 2.** Convergence of stochastic gradient descent for the Gaussian process posterior mean, in terms of training and test error, along with Euclidean error for the representer weights
{% end %}

From this plot, it is clear that stochastic gradient descent does not converge approximately to the correct representer weights.
This lack of convergence is not an artifact of measuring approximation error in representer weight space: it also does not occur under the respective reproducing kernel Hilbert space norm between functions.
In spite of this, test error decreases---in fact, it does so approximately monotonically, in contrast with conjugate-gradient-based approaches for which convergence is heavily non-monotonic.

On the surface, this state of affairs seems rather strange: SGD has essentially failed to solve the linear system we need to solve, therefore failed to compute the correct posterior, yet it produces good test predictions and, as shown previously in one dimension, error bars that are very close to those of the correct Gaussian process.
Let's try to understand this further.

## Implicit bias due to non-convergence via spectral basis functions

To understand why SGD performs well even in cases where it does not converge, let's measure error in a different way.
In the paper, we define the set of *spectral basis functions* `$u^{(i)}(\cdot)$`, which are certain weighted linear combinations of canonical basis functions `$k(x_j,\cdot)$` defined by the kernel, and involve the eigendecomposition of the kernel matrix. 
These are the same functions that arise in kernel principal component analysis.
Let's consider how SGD behaves in terms of the subspace defined by each spectral basis function one-at-a-time.

{% figure(alt=["Convergence in terms of spectral basis functions"] src=["eigenfunctions_and_error.svg"] dark_invert=[true]) %}
**Figure 3.** A comparison of standard large-scale Gaussian process techniques, including conjugate gradients and sparse Gaussian processes with variational inference, under both asymptotics.
{% end %}

From this, we see what is going on: posterior approximation error, measured in terms of the marginal Wasserstein distance at each point, is highest in the regions corresponding to spectral basis functions corresponding to small eigenvalues.
We can use this to understand non-convergence of SGD, by considering three regions of state space:
1. *The interpolation region.* Stochastic gradient descent converges fastest in the directions corresponding to large-eigenvalue spectral basis functions. 
One can show that these large-eigenvalue spectral basis functions, in turn, concentrate in data-dense regions of state space.
Therefore, SGD will produce posterior samples that are correct in regions of state space which are sufficiently data-dense.
2. *The prior region.* Since spectral basis functions are sums of canonical basis functions, if we have a kernel that decays with distance, it follows that approximation eventually disappears as one moves away from the data.
Therefore, for most kernels, SGD will produce posterior samples that are correct sufficiently far from the data.
3. *The exrapolation region.* By elimination, the approximation caused by non-convergence of SGD error can be large in-between the other two regions. Empirically, if we initialize the initial representer weights to be zero, SGD produces error bars which are slightly larger than those of the correct posterior, reverting to the prior slightly quicker than the true posterior would.

A more formal version of these ideas is given in the paper.
With this understanding, let's see how SGD performs in practice on a representative set of problems.

# Experiments

To begin, we perform a set of benchmark comparisons on different datasets between SGD, conjugate gradients, and sparse Gaussian processes trained with variational inference.
For conjugate gradients, we use partial-pivoted-Cholesky-based preconditioning in all cases, except those where it makes performance worse.
We compare these algorithms in terms of test error and negative log-likelihood.
These include small datasets `$N$` in the tens of thousands, and a large dataset with `$N$` around 2 million.
Details are given in the paper.

From the results, shown below, we see that performance significantly depends on noise.
In the standard regime, conjugate gradients tends to performs the best until a certain point when SGD starts to work better.
In the low-noise regime, in contrast, SGD reliably outperforms conjugate gradients and sparse Gaussian processes.
This shows that SGD, in spite of being quite dissimilar from standard approaches, can be a competitive algorithm.

{% figure(alt=["Performance comparison: table"] src=["comparison_table.svg"] dark_invert=[true]) %}
**Table 1.** Regression benchmarks, with `$N$` ranging up to approximately 2 million. The superscript `$(\cdot)^\dagger$` denotes the setting where the likelihood noise is set to a low value of `$\sigma^2 = 10^{-6}$`.
{% end %}

{% figure(alt=["Performance comparison: training curves"] src=["rmse_llh_trace.svg"] dark_invert=[true]) %}
**Figure 4.** Training curves for the regression benchmarks, for a subset of the considered datasets.
{% end %}

## Bayesian optimization

The preceding experiment shows that SGD can be competitive in terms of predictive performance: next, we evaluate how well it behaves in terms of uncertainty.
We do so by way of a large-scale synthetic Bayesian optimization benchmark: we sample a set of random target functions from the prior, pre-train a Gaussian process with `$N = 50,000$` data points, then perform Bayesian optimization via parallel Thompson sampling.
Results are given below, for two different compute budgets. 
From these, we see that SGD either matches or outperforms both baselines, depending on how much compute budget these methods are given.
This suggests the benign non-convergence, which we previously saw in one dimension and examined theoretically, carries over to decision-making settings.

{% figure(alt=["Thompson sampling comparison"] src=["thompson.svg"] dark_invert=[true]) %}
**Figure 5.** Parallel thompson sampling benchmark, for two different computational budgets.
{% end %}

# Conclusion

In this, we explored using stochastic gradient descent to approximately compute Gaussian process posteriors, by way of means and function samples.
We examined how to derive appropriate stochastic optimization objectives for doing so, and showed that SGD can produce accurate predictions even in cases where it does not converge to the respective optimum under the given compute budget.
We developed a spectral characterization of the effect of non-convergence in terms of the spectral basis functions.
We showed that, on a Thompson sampling benchmark where well-calibrated uncertainty is critical, SGD matches or exceeds the performance of more computationally expensive baselines.



# References


[^cg]: J. Gardner, G. Pleiss, K. Q. Weinberger, D. Bindel, and A. G. Wilson. GPyTorch: Blackbox Matrix-matrix Gaussian Process Inference with GPU Acceleration. NeurIPS 2018.

[^ip-s]: J. Hensman, N. Fusi, and N. D. Lawrence. Gaussian processes for big data. UAI 2013.

[^ip-v]: M. Titsias. Variational learning of inducing variables in sparse Gaussian processes. AISTATS 2009.

[^ip-theory]: One can show theory that backs up both of these results: inducing points are, under appropriate assumptions, scale approximately linearly under infill asymptotics,[^ip-conv] and conjugate-gradient-based Gaussian processes are sub-cubic under large-domain asymptotics.[^num-stability]

[^efficient-sampling]: J. T. Wilson, V. Borovitskiy, A. Terenin, P. Mostowsky and M. P. Deisenroth. Efficiently Sampling Functions from Gaussian Process Posteriors. ICML 2020.

[^pathwise-conditioning]: J. T. Wilson, V. Borovitskiy, A. Terenin, P. Mostowsky and M. P. Deisenroth. Pathwise Conditioning of Gaussian Processes. JMLR 2021.

[^ip-conv]: D. R. Burt, C. E. Rasmussen, and M. van der Wilk. Rates of Convergence for Sparse Variational Gaussian Process Regression. ICML 2019.

[^num-stability]: A. Terenin, D. R. Burt, A. Artemev, S. Flaxman, M. van der Wilk, C. E. Rasmussen, and H. Ge. Numerically Stable Sparse Gaussian Processes via Minimum Separation using Cover Trees. JMLR 2023.

 
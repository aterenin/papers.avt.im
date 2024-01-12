+++
title = "Efficiently Sampling Functions from Gaussian Process Posteriors"
[extra]
authors = [
    {name = "James T. Wilson", star = true},
    {name = "Viacheslav Borovitskiy", star = true},
    {name = "Alexander Terenin", url = "https://avt.im/", star = true},
    {name = "Peter Mostowsky", star = true},
    {name = "Marc Peter Deisenroth", url = "https://deisenroth.cc/"},
]
star = "Equal contribution"
venue = {name = "ICML", date = 2020-07-13, url = "https://icml.cc/Conferences/2020"}
buttons = [
    {name = "Paper", url = "https://proceedings.mlr.press/v119/wilson20a.html"},
    {name = "PDF", url = "https://arxiv.org/pdf/2002.09309"},
    {name = "Code", url = "https://github.com/j-wilson/GPflowSampling"},
    {name = "Video", url = "https://icml.cc/virtual/2020/poster/6461"},
]
katex = true
large_card = true
+++

Gaussian processes (GPs) play a pivotal role in many complex machine learning algorithms. For example, sequential decision-making strategies such as Bayesian optimization frequently use GPs to represent different actions' possible outcomes. 
Actions are then chosen by maximizing the conditional expectation of a chosen reward functional with respect to the GP posterior. 
These expectations quickly become intractable when dealing with more expressive reward functions, but may be efficiently estimated via Monte Carlo methods.

Sampling from GP posteriors is usually accomplished using location-scale transforms.
Given a set of points `$\mathbf{X}_{*} = \{\boldsymbol{x}_{1}, \ldots, \boldsymbol{x}_{*}\}$` at which to sample, we first compute the conditional distribution `$\boldsymbol{f}_{* \mid \boldsymbol{u}} \sim \mathcal{N}(\boldsymbol{m}_{* \mid \boldsymbol{u}}, \mathbf{K}_{*, * \mid \boldsymbol{u}})$` of `$\boldsymbol{f}_{*} = f(\mathbf{X}_{*})$` given `$m$` observations `$\boldsymbol{u} = f(\mathbf{Z})$`.
We then use the mean and covariance of this Gaussian conditional to draw samples. While this generative procedure is relatively error-free, it incurs cubic complexity.
Below, we detail a novel sampling scheme based off of *pathwise conditioning*: rather than conditioning the prior as a distribution, we update the prior as realized in terms of sample paths. This approach conveys a number of immediate advantages.

  - Its complexity is *linear* in the number of test points.
  - It yields an actual *function draw* that we may freely evaluate and differentiate anywhere.
  - Its discretization error is easy to understand and control.

The end result of this process, _efficient sampling_, uses the stengths of location-scale methods to counteract the weaknesses of popular Fourier-feature-based alternatives, and vice-versa.


# Pathwise updates for Gaussian process posteriors


A Gaussian process is a distribution over functions `$f : \mathcal{X} \to \mathbb{R}$` with Gaussian marginals, meaning that `$\boldsymbol{f}_{*}$` is multivariate normal for any finite set of locations.
Given observations `$\boldsymbol{u} = f(\mathbf{Z})$`, the ensuing posterior is typically portayed as a Gaussian distribution `$\boldsymbol{f}_{*} \mid \boldsymbol{u} \sim \mathcal{N}(\boldsymbol{m}_{* \mid \boldsymbol{u}}, \mathbf{K}_{*, * \mid \boldsymbol{u}})$` with centered moments

```
$$
\begin{aligned}
\boldsymbol{m}_{* \mid \boldsymbol{u}} &= \mathbf{K}_{*, m}\mathbf{K}_{m,m}^{-1}\boldsymbol{u}
&
&\quad
&
\mathbf{K}_{*,* \mid \boldsymbol{u}} &= \mathbf{K}_{*, *} - \mathbf{K}_{*, m} \mathbf{K}_{m, m}^{-1} \mathbf{K}_{m, *}
.
\end{aligned}
$$
```

This way of writing Gaussian posteriors mirrors the standard way of thinking about them: in terms of mean vectors and covariance matrices.[^gpml] 
A less familiar but equally valid way of expressing Gaussian conditionals is given by *Matheron's rule*: if `$\boldsymbol{a}, \boldsymbol{b}$` are jointly Gaussian, then

```
$$
(\boldsymbol{a} \mid \boldsymbol{b}=\boldsymbol{\beta}) \stackrel{\operatorname{d}}{=} \boldsymbol{a} + \operatorname{Cov}(\boldsymbol{a},\boldsymbol{b})\operatorname{Cov}(\boldsymbol{b},\boldsymbol{b})^{-1}(\boldsymbol{\beta} - \boldsymbol{b}).
$$
```

Accordingly, we may sample `$\boldsymbol{a} \mid \boldsymbol{b} = \boldsymbol{\beta}$` by updating a joint draw `$\boldsymbol{a}, \boldsymbol{b}$` to account for the residual `$\boldsymbol{\beta} - \boldsymbol{b}$`, thereby inducing a corresponding change in `$\boldsymbol{a}$` by virtue of its covariance with `$\boldsymbol{b}$`. 
This procedure is illustrated below for the simple case of bivariate normal random variables.


{{ figure(alt = ["Sample from the joint distribution","Transform the samples into the conditional distribution"], src = ["joint.svg","cond.svg"], subcaption = ["(1) Jointly sample `$\boldsymbol{a},\boldsymbol{b}$`","(2) Transform `$\boldsymbol{a},\boldsymbol{b}$` into `$\boldsymbol{a}\mid\boldsymbol{b}$`"], dark_invert = [true,true]) }}


Extending this concept to Gaussian process priors `$f \sim \mathcal{GP}(0, k)$` leads to a pathwise characterization of their posteriors, namely

```
$$
(f \mid \boldsymbol{u})(\cdot) \stackrel{\operatorname{d}}{=} f(\cdot) + \mathbf{K}_{(\cdot), m} \mathbf{K}_{m,m}^{-1} (\boldsymbol{u} - f(\mathbf{Z})).
$$
```

As in finite dimensional cases, we may sample from the posterior via pathwise updating of draws from the prior.
For sparse GPs, this process involves generating a separate draw from the inducing distribution `$q(\boldsymbol{u})$`.
In addition to specifying an update rule, this pathwise representation decomposes GP posteriors as dependent sums of prior and update terms. 
Examining both terms, we see that prior and update, respectively, scale in cubic and linear fashion with the number of test points. 
Efficiently sampling from the posterior therefore reduces to efficiently sampling from the prior.


# Efficiently sampling from the prior
Different choices of prior afford different avenues for fast sampling. 
Here, we focus on the standard setting of stationary kernels `$k(\boldsymbol{x}, \boldsymbol{x}^\prime) = k(\boldsymbol{x} - \boldsymbol{x}^\prime)$`. In such cases, the kernel can be viewed an inner product in a reproducing kernel Hilbert space `$\mathcal{H}$`, and approximated by random Fourier features[^rff] such that

```
$$
k(\boldsymbol{x},\boldsymbol{x}') = \langle\boldsymbol{\varphi}(\boldsymbol{x}),\boldsymbol{\varphi}(\boldsymbol{x}')\rangle_{\mathcal{H}} \approx \boldsymbol{\phi}(\boldsymbol{x})^\top \boldsymbol{\phi}(\boldsymbol{x})
$$
```

where `$\varphi : \mathcal{X} \to \mathcal{H}$` is a feature map and `$\boldsymbol{\phi} : \mathcal{X} \to \R^\ell$` is an `$\ell$`-dimensional approximation thereof.[^rffeq] If `$\boldsymbol{w} \sim \mathcal{N}(\boldsymbol{0}, \mathbf{I})$`, then the right-hand side of

```
$$
f(\cdot) \approx \boldsymbol{\phi}(\cdot)^\top \boldsymbol{w} = \sum_{i=1}^\ell w_i \phi_i(\cdot)
$$
```

is a random function with Gaussian marginals whose covariance is approximately `$k(\cdot, \cdot)$`. 
This means that realizations of this function are draws from an approximate GP prior. 
Unlike location-scale methods, this approximate draw is an actual function and exhibits linear time complexity in the number of test locations. 
Moreover, the error introduced by random Fourier feature methods is well-understood[^rffe] and controlled by the number of basis functions `$\ell$` used in the approximation.


# Efficient sampling
Putting these ideas together, we obtain the _efficient sampling_ approximation

```
$$
(f\mid\boldsymbol{u})(\cdot) \approx \sum_{i=1}^\ell w_i \phi_i(\cdot) +  \sum_{j=1}^m v_j k(\cdot, \boldsymbol{z}_j)
$$
```

where `$\boldsymbol{v} = \mathbf{K}^{-1}_{m,m}(\boldsymbol{u} - \mathbf{\Phi}\boldsymbol{w})$` is defined in terms a feature matrix `$\mathbf{\Phi}$` with rows `$\phi_i(\mathbf{Z})$`. 
Here, we have chosen to explicitly represent the update as a sum over canonical basis functions `$k(\cdot, \boldsymbol{z}_j)$` to further emphasize that efficient sampling produces function draws.
This is visualized below.


{{ figure(alt = ["The prior term and its Fourier basis functions","The update term and its canonical basis functions"], src = ["prior.svg","data.svg"], subcaption = ["The prior term and its Fourier basis functions","The update term and its canonical basis functions"], dark_invert = [true,true]) }}


Unlike previous approximate GPs, this approach is specifically tailored for sampling. Just as the Fourier basis excells at representing the prior, the canonical basis excells at representing the data.[^svgpr]
Hence, using Matheron's rule to separate out prior from update allows us obtain the best of both worlds by utilizing a suitable basis for each term.


# Takeaways
Efficient sampling is a general-purpose technique for efficiently drawing functions from GP posteriors. 
In addition to use cases outlined above, this technique can be employed as a plug-in approach to sampling from many common types of GP posteriors, such as those arising from sparse approximations or Gaussian observations.
These expressions are given in the paper.
Together with its ease-of-use and pathwise differentiability, efficient sampling's linear time complexity makes it an ideal choice for GP-based Monte Carlo methods.


# References

[^gpml]: C. E. Rasmussen and C. K. Williams. Gaussian Processes for Machine Learning. 2006.

[^rff]: A. Rahimi and B. Recht. Random features for large-scale kernel machines. NeurIPS, 2008.

[^rffeq]: Using the random Fourier feature method, we set `$\phi_i(\cdot) = \sqrt{2/\ell}\cos(\boldsymbol\theta_i^T(\cdot) + \tau_i)$`, where `$\boldsymbol\theta_i$` are drawn from the kernel's spectral measure, and `$\tau_i \sim \mathcal{U}(0,2\pi)$`.

[^rffe]: D. J. Sutherland and J. Schneider. On the error of random fourier features. UAI, 2015.

[^svgpr]: D. R. Burt, C. E. Rasmussen, and M. van der Wilk. Rates of convergence for sparse variational Gaussian process regression. ICML, 2019.

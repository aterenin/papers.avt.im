+++
title = "Stochastic Poisson Surface Reconstruction with One Solve using Geometric Gaussian Processes"
[extra]
authors = [
    {name = "Sidhanth Holalkere", url = "https://sholalkere.github.io"},
    {name = "David Bindel", url = "https://www.cs.cornell.edu/~bindel/"},
    {name = "Silvia Sellán", url = "https://www.silviasellan.com/"},
    {name = "Alexander Terenin", url = "https://avt.im/"},
]
venue = {name = "ICML", date = 2025-07-13, url="https://icml.cc/"}
buttons = [
    {name = "Paper", url = "https://openreview.net/forum?id=SUX6Wzy9t1"},
    {name = "PDF", url = "https://arxiv.org/pdf/2503.19136"},
    {name = "Code", url = "https://github.com/sholalkere/GeoSPSR"},
]
katex = true
large_card = false
favicon = false
+++


{{ figure(src=["scorp.svg"], dark_invert=[true], alt=["Statistical queries such as in probability computed by our method."], body="Figure 1: An illustration of stochastic Poisson surface reconstruction.") }}

Surface reconstruction algorithms are used to convert point cloud data—the most common format in which real-world scans are captured—into other downstream-usable formats, such as meshes or other shape representations. 
For an orientable surface `$\Omega \subseteq \mathbb{R}^3$`, they work by taking a set of oriented points `$(x_i, v_i)_{i=1}^N$` from the surface, where `$v_i$` are normal vectors, and attempt to produce a suitable shape representation of `$\Omega$`.
In this work, we study a class of *uncertainty quantification* algorithms which produce a probability distribution over possible surfaces, and develop an improved numerical pipeline for performing the required calculations for doing so.

# Uncertainty Quantification for Surface Reconstruction

To begin, we first briefly review *Poisson surface reconstruction*—the most widely-used surface reconstruction algorithm, before describing how to quantify its uncertainty.
Poisson surface reconstruction reconstructs the surface as an implicit function `$f$`, producing an `$f$` which is negative on the inside of the surface, `$0$` on the surface, and positive on the outside of the surface. 
The algorithm does so by creating a dense, global mesh encapsulating the input points, and then using finite elements to solve the *Poisson equation*, namely the partial differential equation `$\Delta f(x) = \nabla \cdot v(x)$`, on this mesh.
This is most often done using finite elements.

## Stochastic Poisson Surface Reconstruction

When observations are sparse, surface reconstruction becomes an under-determined problem.
Instead of reasoning about *one* surface, we may want to consider the *distribution* of surface the input might have been sampled from. 
To achieve this, [Sellán and Jacobson](https://dl.acm.org/doi/10.1145/3550454.3555441) introduce a statistical extension called *stochastic Poisson surface reconstruction*, which adds these capabilities. 
Instead of a deterministic `$v$`, they place a Gaussian process prior `$v \sim \operatorname{GP}(0, k)$` and apply Bayesian learning to compute the conditional distribution of `$f \mid v$`, which is also a Gaussian process. 
This works because the Poisson solve is linear, and Gaussian processes are closed under linear transformations.
Figure 1 gives an illustration of these computations.

By its Bayesian nature, stochastic Poisson surface reconstruction quantifies uncertainty.
However, due to the same nature, the computational pipeline of [Sellán and Jacobson](https://dl.acm.org/doi/10.1145/3550454.3555441) inherits and introduces a few limitations:

1. It requires one to work with a global scene is therefore not output-sensitive: one cannot obtain uncertainty for a part of the surface without first obtaining it for the whole surface.
2. It introduces approximations which make covariance calculations inaccurate for realistic length scales.
3. It couples the discretization structure and the prior's kernel length scale, which can cause runtime to scale exponentially with reconstruction resolution.

Our key contribution is to reduce or remove these limitations, by reformulating the problem using geometric Gaussian processes.

# Avoiding the Poisson Solve using Geometric Gaussian Processes

To alleviate the preceding limitations, our approach will be merge the two parts of the computational pipeline used by [Sellán and Jacobson](https://dl.acm.org/doi/10.1145/3550454.3555441): (1) the Gaussian process interpolation on `$v$` and (2) the Poisson solve used to produce the implicit surface `$f$` from `$v$`.
To see how, note first that Gaussian processes are closed under linear transformations, so we can directly express the posterior using pathwise conditioning as
```
$$
\underset{\mathclap{\substack{\text{posterior}\\\text{over solution}}}}{\undergroup{(f\mid\boldsymbol{v})(\cdot)}} \quad=\quad \underset{\mathclap{\substack{\text{prior over}\\\text{solution}}}}{\undergroup{f(\cdot)}} \quad+\quad\underset{\mathclap{\substack{\text{cross-covariance}\\\text{from Poisson eqn.}}}}{\undergroup{\mathbf{K}_{f(\cdot)\boldsymbol{v}}}}\quad\underset{\mathclap{\text{standard GP solve}}}{\undergroup{\vphantom{\mathbf{K}_{f(\cdot)\boldsymbol{v}}}(\mathbf{K}_{\boldsymbol{v}\boldsymbol{v}}+\mathbf\Sigma)^{-1}(\boldsymbol{v} - v(\boldsymbol{x}) - \boldsymbol\varepsilon)}}
.
$$
```
These expressions involve matrices generated from the cross-covariance `$k_{f,v}$` and the scalar field covariance `$k_f$`: it is not immediately clear how to derive these from the vector field covariance `$k_v$`.
Our paper also contains expressions for the posterior mean and covariance, which are similar.

To compute these, we will assume periodic boundary conditions, along with a periodic kernel, and apply the theory of geometric Gaussian processes. 
Note that these are assumed on the *bounding box surrounding the surface*, and not the surface itself.
Our main result is as follows.
Let `$k_v$` be a product kernel `$k_v(x, x') = \prod_{i=1}^d k_{v_i}(x, x')$` where each component has a Mercer expansion `$k_{v_i}(x, x ') = \sum_{n \in \mathbb{Z}^d} \rho_{v_i}(n)(\sin(\langle n, x \rangle) \sin(\langle n, x ' \rangle) + \cos( \langle n, x \rangle) \cos(\langle n, x ' \rangle))$`.
Note that this expansion is known for periodic Matérn kernels, where `$\rho_{v_i}$` takes is an explicit analytic function.
Under these assumptions, we show that the covariance between the interpolated vector field and Poisson equation's solution is
```
$$
k_{f, v_i}(x, x ') = \sum_{\substack{n \in \mathbb{Z}^d\\n\neq0}} \frac{n_i \sqrt{\rho_{v_i}(n)}}{\|n\|^2} \sin(n \cdot (x - x'))
$$
```
and that the covariance of the Poisson equation's solution is
```
$$
k_f(x, x') = \sum_{\substack{n \in \mathbb{Z}^d\\n\neq0}} \frac{\sum_{i=1}^d n_i^2 \rho_{v_i}(n)}{\|n\|^4} (\sin(\langle n, x \rangle) \sin(\langle n, x ' \rangle) + \cos( \langle n, x \rangle) \cos(\langle n, x ' \rangle)).
$$
```
This allows us to evaluate the posterior without any finite element solves, by truncating these sums!
We also include sampling formulas in the paper, and describe strategies for selecting the truncation level and amortizing its cost.


### Demonstrations and Examples

We now illustrate some example surface reconstruction applications common in computer graphics, which can be handled using our approach.
This is shown in Figure 2, 3, 4, and 5 below.

{{ figure(src=["collision.svg"], dark_invert=[true], alt=["Collision probabilities of various cats and a dragon."], body="Figure 2: Collision detection.") }}

{{ figure(src=["falcon.svg"], dark_invert=[true], alt=["Posterior samples along a ray looking at a falcon."], body="Figure 3: Ray casting.") }}

{{ figure(src=["sampling.svg"], dark_invert=[true], alt=["Samples of the posterior given a partial scan of a bunny."], body="Figure 4: Monte Carlo samples from the reconstructed surface' posterior.") }}

{{ figure(src=["dragon_lengthscale.svg"], dark_invert=[true], alt=["Runtime of our method and prior method for a variety of length scales."], body="Figure 5: Runtime as a function of length scale for our approach, and the original two-stage stochastic Poisson surface reconstruction pipeline, where we see that our runtime does not depend significantly on the kernel's length scale, compared to prior work.") }}


# Conclusion

We developed an approach for avoiding recursive linear solves in stochastic Poisson surface reconstruction, which was achieved by applying periodic boundary conditions and working with geometric Gaussian processes.
The proposed method was shown to support the same set of statistical queries as prior work, and additionally provide new, random-sampling-based queries. 
These are able to take into account smoothness properties captured by the kernel's correlations, which are lost as a result of diagonal kernel matrix approximations made in prior work.
Our work also constitutes a first step to incorporating sample-efficient data acquisition techniques from the Gaussian process and Bayesian optimization literatures into surface reconstruction and computer graphics.

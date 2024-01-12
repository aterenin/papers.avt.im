+++
title = "Matérn Gaussian Processes on Graphs"
[extra]
authors = [
    {name = "Viacheslav Borovitskiy", star = true},
    {name = "Iskander Azangulov", star = true},
    {name = "Alexander Terenin", url = "https://avt.im/", star = true},
    {name = "Peter Mostowsky"},
    {name = "Marc Peter Deisenroth", url = "https://deisenroth.cc/"},
    {name = "Nicolas Durrande", url = "https://sites.google.com/site/nicolasdurrandehomepage/"},
]
venue = {name = "AISTATS", date = 2021-04-13, url = "https://aistats.org/aistats2021/"}
buttons = [
    {name = "Paper", url = "https://proceedings.mlr.press/v130/borovitskiy21a.html"},
    {name = "PDF", url = "https://arxiv.org/pdf/2010.15538"},
    {name = "Code", url = "https://github.com/spbu-math-cs/Graph-Gaussian-Processes"},
    {name = "Poster", url = "/presentations/2021-03-21-Graph-Matern-GP-Poster/2021-03-21-Graph-Matern-GP-Poster.pdf"},
    {name = "Video", url = "https://virtual.aistats.org/virtual/2021/oral/1817"},
]
katex = true
large_card = true
+++

Gaussian processes are a model class for learning unknown functions from data.
They are particularly of interest in statistical decision-making systems, due to their ability to quantify and propagate uncertainty.
In this work, we study analogs of the popular Matérn class where the domain of the Gaussian process is replaced by a weighted undirected graph.
We focus particularly on connecting prior work originally developed for applications such as geostatistics with the modern machine learning toolkit, with emphasis on automatic differentiation.
This facilitates the use of graph Matérn Gaussian processes in Bayesian optimization and other Gaussian process application areas.


# Matérn Gaussian processes

One of the most well-known kernels is the Matérn kernel, which is defined as

```
$$
k(x,x') = \sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)} \left(\sqrt{2\nu} \frac{\Vert x-x' \Vert}{\kappa}\right)^\nu K_\nu \left(\sqrt{2\nu} \frac{\Vert x-x' \Vert}{\kappa}\right)
$$
```

where `$\sigma^2,\kappa,\nu$` are the variance, length scale, and smoothness parameters, and `$K_\nu$` is the modified Bessel function of the second kind.
One of this kernel's key properties of interest to us is that the Gaussian process it induces satisfies the stochastic partial differential equation

```
$$
\left(\frac{2 \nu}{\kappa^2} - \Delta\right)^{\frac{\nu}{2} + \frac{d}{4}}f = \mathcal{W}
$$
```

where `$\Delta$` is the Laplacian, and `$\mathcal{W}$` is a white noise Gaussian process.[^lgps]

In the Euclidean setting, this kernel is often one of the first choices practitioners consider because of its simple form and easy-to-understand hyperparameters.
There, it defines a Gaussian process `$f : \mathbb{R}^d \to \mathbb{R}$`.
The goal of this work is to explore generalizations of this expression to the case where we are interested in a Gaussian process `$f : G \to \mathbb{R}$`, where `$G$` is the set of nodes of a weighted undirected graph.
Below, we visualize these processes, for a sequence of weighted graphs approaching a sphere in the limit.


{{ figure(alt = ["Graph Matérn Gaussian processes: sphere"], src = ["convergence.svg"])}}


# Graph Matérn Gaussian processes

To begin, we consider the SPDE characterization of Matérn Gaussian processes, and, following prior work in the statistics community,[^gmrf] generalize them to the graph setting by replacing the left-hand-side and right-hand-side of the SPDE with appropriate graph-theoretic notions.
Specifically, we replace the Euclidean Laplacian `$\Delta$` by the graph Laplacian `$\mathbf\Delta$`, and the white noise by a standard Gaussian.
This gives the equation

```
$$
\left(\frac{2 \nu}{\kappa^2} + \mathbf\Delta\right)^{\frac{\nu}{2}} \boldsymbol{f} = \mathcal{W}\hspace*{-2.42ex}\mathcal{W}\hspace*{-2.42ex}\mathcal{W}
$$
```

where we have replaced `$-\Delta$` with `$\mathbf\Delta$` to reflect the sign convention for the Laplacian commonly used in graph theory,[^sign] dropped the `$d/4$` term because it is unneeded, and replaced the white noise Gaussian process with an IID Gaussian.
To make sense of this equation, we need to know what it means to raise a matrix such as `$\mathbf\Delta$` to a potentially non-integer power.
Here, we employ an idea known as *functional calculus*: we define `$\left(\frac{2 \nu}{\kappa^2} + \mathbf\Delta\right)^{\frac{\nu}{2}}$` to be the matrix obtained by applying the transformation `$\lambda\mapsto\left(\frac{2 \nu}{\kappa^2} + \lambda\right)^{\frac{\nu}{2}}$` to each of the *eigenvalues* of `$\mathbf\Delta$`,[^fc] while keeping the eigenvectors unchanged.
Rearranging the above equation gives

```
$$
\boldsymbol{f} \sim \operatorname{N}\left(\left(\frac{2 \nu}{\kappa^2} + \mathbf\Delta\right)^{-\nu}\right)
$$
```

as a multivariate Gaussian distribution over the graph's edges, where, again, both addition and exponentiation are defined in the sense of functional calculus.
We refer to this multivariate Gaussian as a *graph Matérn Gaussian process* to reflect that its covariance reflects the structure of the graph.

# Properties

In our work, we review and summarize a number of properties of graph Matérn Gaussian processes, as well as techniques that can be used to train them.
Some of these are listed below.

1. *Training via inducing points.* Graph Matérn Gaussian processes can be trained using stochastic variational inference via inducing points, which enables their use in mini-batch and non-conjugate settings, and allows them to be trained using modern toolkits based on automatic differentiation.
2. *Fourier features.* The graph Laplacian can be eigenfactorized, which enables one to define graph Fourier features, which accelerate computation and can enable graph Matérn Gaussian processes to be approximated even in cases where the graph is too large to fit in memory.
3. *Sparsity.* For sparse graphs, `$\mathbf\Delta$` is sparse, and hence for many graphs the precision matrices `$\left(\frac{2 \nu}{\kappa^2} + \mathbf\Delta\right)^\nu$` are sparse. Following prior work on Gaussian Markov random fields, this can be exploited to accelerate computation.
4. *Non-uniform variance.* The prior variance of the graph Matérn kernel can be non-uniform in a way that reflects the structure of the graph.
5. *Limits.* We review connections with other graph kernels, as well as with Matérn kernels on Riemannian manifolds,[^gprm] such as the sphere illustrated previously.

In total, these perspectives provide a number of useful ways to look at and understand graph Matérn Gaussian processes, and to deploy them using modern automatic-differentiation-based machine learning tools.

# Example: traffic data

To conclude, we provide an illustrative demonstration on what kind of datasets graph Matérn kernels can be applied to.
Here, we consider graph interpolation of traffic data from the *California Performance Management System*.
Nodes for which data is available are indicated with white circles.
Global and local means and standard deviations for this dataset can be seen below.


{{ figure(alt = ["Traffic data: global mean","Traffic data: global standard deviation"], src = ["global_mean.svg","global_std.svg"], dark_invert = [true,true], subcaption = ["(a) Mean: global","(b) Standard deviation: global"]) }}


{{ figure(alt = ["Traffic data: local mean","Traffic data: local standard deviation"], src = ["junction_mean.svg","junction_std.svg"], dark_invert = [true,true], subcaption = ["(a) Mean: local","(b) Standard deviation: local"]) }}


From the above, we see that the predictions made by the graph Matérn Gaussian process reflects the structure of the graph. 
In particular, posterior standard deviation increases as we move away from nodes which have data.
Additionally, predictions on different sides of the road---which are represented by different nodes and vertices in the graph---can differ, even though these locations are located very close in Euclidean distance.
While this example is not entirely realistic---note that we consider pure graph interpolation, and do not introduce time-dependence of any kind---we believe that it provides a good illustration of what kind of models are possible, which we hope will stimulate the imaginations of practitioners.

# Concluding remarks

We present techniques for working with graph Matérn Gaussian processes, with a focus on synthesizing and connecting various prior perspectives in the literature.
We do so through the unifying lens of the graph Laplacian, which enables us to define and efficiently approximate graph Matérn Gaussian processes in a different ways that can be tailored to suit the application of interest.
This includes use of sparsity, following ideas in the Gaussian Markov random field literature, and the use of Fourier feature expansions, which have become popular tools in the Gaussian process and kernel methods communities.
We hope these perspectives enable a wider array of machine learning practitioners to leverage these models to apply Bayesian optimization and other techniques to novel settings.



# References

[^lgps]: M. Lifshits. Lectures on Gaussian Processes. Springer, 2012.

[^gmrf]: H. Rue and L. Held. Gaussian Markov random fields: theory and applications. CRC Press, 2005.

[^sign]: This different sign convention should always be double-checked. Our paper's original draft contained an error in one of our baseline experiments precisely due to accidental use of the wrong sign.

[^fc]: Note that, following the notation of functional calculus, the addition operation in `$\left(\frac{2 \nu}{\kappa^2} + \mathbf\Delta\right)^{\frac{\nu}{2}}$` is performed to the eigenvalues of `$\mathbf\Delta$`, just like the exponentiation: this is not to be confused with element-wise addition.

[^gprm]: V. Borovitskiy, P. Mostowsky, A. Terenin, and M. P. Deisenroth. Matérn Gaussian processes on Riemannian Manifolds. NeurIPS, 2020.
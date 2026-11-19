+++
title = "A Bayesian Approach for Task-Specific Next-Best-View Selection with Uncertain Geometry"
[extra]
authors = [
    {name = "Jingsen Zhu", url = "https://jingsenzhu.github.io/"},
    {name = "Silvia Sellán", url = "https://www.cs.columbia.edu/~silviasellan/"},
    {name = "Alexander Terenin", url = "https://avt.im/"},
]
venue = {name = "SIGGRAPH", date = 2026-07-19, url = "https://s2026.siggraph.org/"}
buttons = [
    {name = "Paper", url = "https://arxiv.org/abs/2605.05095/"},
    {name = "PDF", url = "https://arxiv.org/pdf/2605.05095"},
    {name = "Code", url = "https://github.com/jingsenzhu/BayesianNBV/"}
]
large_card = false
favicon = false
+++

{% <figure src={["teaser.svg"]} dark_invert={[true]} alt={["Task-specific applications enabled by our framework."]}> %}
**Figure 1**: *Task-specific* applications enabled by our framework for optimizing the next view angle given an incomplete observation of a given object.
{% </figure> %}

**Abstract.** We develop a framework for task-specific active next-best-view selection in 3D reconstruction from point clouds, by casting the problem in the language of Bayesian decision theory. Our framework works by (a) placing a prior distribution over the space of implicit surfaces, (b) using recently-developed stochastic surface reconstruction methods to calculate the resulting posterior distribution, then (c) using the posterior distribution to carefully reason about which view to scan next. This enables us to perform camera selection in a manner that is directly optimized for the intended use of the reconstructed data -- meaning, we reduce uncertainty only in those regions that make a difference in the task at hand, as opposed to prior approaches that reduce it uniformly across space. We evaluate our method across three distinct downstream tasks: semantic classification, segmentation, and PDE-guided physics simulation. Experimental results demonstrate that our framework achieves superior task performance with fewer views compared to commonly used baselines and prior general uncertainty-reduction techniques.

# Next-Best-View Selection

We begin by briefly describing the task we are trying to solve.
Let `$f : [0,1]^3 \to [-1,1]$` be an implicit surface representation of the unknown scene.
We assume that information about `$f$` is acquired through a *scanning operation* `$\mathrm{scan}(\cdot;f) : SE(3) \to \mathbb{D}$`, where `$\theta\in\mathrm{SE}(3)$` represents camera angles, and `$\mathbb{D}$` is a space representing oriented point clouds. 
More precisely, `$(\boldsymbol{x},\boldsymbol{n}) \in \mathbb{D}$` consists of a dataset of surface locations `$\boldsymbol{x}$`, and surface normals `$\boldsymbol{n}$`, of matched but otherwise potentially variable length.
We assume the unknown geometry `$f$` is only accessible in a black-box manner through these discrete observations, to mimic 3D scanning applications without explicit prior knowledge assumed about the object.
At each time step `$t$`, we have a set of previously selected viewpoints `$\theta_\tau$`, `$\tau=1,..,t$` and the resulting cumulative dataset `$\mathcal{D}_t = \bigcup_{\tau=1}^t \mathrm{scan}(\theta_\tau;f)$`, where the union symbol denotes concatenation of observations.
Our objective is to carefully select a subsequent viewpoint `$\theta_{t+1}$`.

We formulate sequential scanning in a Bayesian manner, by placing a prior distribution `$p(f)$` over the implicit surface.
We can condition this prior on the data to obtain the posterior distribution `$p(f \mid \mathcal{D}_t)$`.
Using this, we can compute *simulated scans* under the posterior distribution, denoted by `$\mathrm{scan}(\cdot;f\mid\mathcal{D}_t)$`.

## Algorithm

With formalizing active point cloud scanning as a Bayesian decision problem, we design our algorithm as the following steps:
1. Given the partially scanned point cloud `$\mathcal{D}_t$` at time step `$t$`, we obtain the posterior geometry reconstruction `$p(f \mid \mathcal{D}_t)$` through uncertain geometry reconstruction methods: specifically, we use Stochastic Poisson Surface Reconstruction throughout this work.[^spsr]
2. We define a *utility function* `$u : \mathbb{D} \to \mathbb{R}$`, which is *task-dependent*: describing how useful the information contained in a partial scan is for the task at hand.
3. By conducting the *simulated* scan using the posterior, we select the next view according to the *expected utility improvement* acquisition function: this provides a score for the potential value of a candidate camera `$\theta$`, and is defined as
```
$$
\alpha^{(u)}_{f\mid\mathcal{D}_t}(\theta) = \mathbb{E} \max{\left(0, u(\mathcal{D}_t \cup \mathrm{scan}(\theta; f\mid\mathcal{D}_t))- u(\mathcal{D}_t)\right)}
.
$$
```
4. We then select the next view by solving the optimization problem 
```
$$
\theta_{t+1} = \operatorname*{\arg\max}_{\theta \in \Theta} \alpha^{(u)}_{f\mid\mathcal{D}_t}(\theta)
.
$$
```

In practice, the expectation in `$\alpha^{(u)}_{f\mid\mathcal{D}_t}$` is estimated by a Monte Carlo estimation from multiple samples from the posterior distribution `$\hat{f} \sim p(f \mid \mathcal{D}_t)$`. 
Please refer to *Algorithm 1* in our paper for pseudocode.

## Task-specific Utility Functions

Using the preceding ideas, we provide a general Bayesian next-best-view selection framework that allows for the tailoring of camera selection to the specific requirements of the downstream application, through the selection of utility and acquisition functions.
Our paper handles the following as examples:
1. *3D point cloud classification*, to identify the correct class label of the underlying object with the minimum number of viewpoints.
2. *3D Semantic Segmentation and Part Discovery*, to discover all semantic parts of the underlying object with minimal observations.
3. *Physics-Informed Scanning*, driven by the physical properties of the reconstructed scene, for instance heat diffusion.
4. *Traditional task-agnostic 3D reconstruction*, which typically aims for scene coverage and global scene uncertainty reduction.

# Results

We evaluate the proposed method in several settings, shown in brief below.

## Classification

{% <figure src={["classification.svg"]} dark_invert={[true]} alt={["Classification results."]}> %}
**Figure 2:** We show a comparison between our method and baseline methods for a classification task. Our approach achieves a more precise classification result with fewer views.
{% </figure> %}

## Segmentation

{% <figure src={["segmentation.mp4"]} alt={["Segmentation results."]} source_attributes={["autoplay muted loop playsinline controls"]}> %}
**Figure 3:** We show a comparison between our method and furthest point sampling (FPS), for an object segmentation task, where the goal is to discover more parts of an object. Our approach discovers more parts with fewer views.
{% </figure> %}


## Heat Diffusion

{% <figure src={["heat.mp4"]} alt={["Heat diffusion results."]} source_attributes={["autoplay muted loop playsinline controls"]}> %}
**Figure 4:** We show a comparison between our method and furthest point sampling (FPS) for a heat diffusion task. With the same number (4) of cameras, our approach achieves a better heat simulation fidelity than the baseline.
{% </figure> %}


# Citation

```
@article{zhu2026bayesian,
  title={A Bayesian Approach for Task-Specific Next-Best-View Selection with Uncertain Geometry},
  author={Zhu, Jingsen and Sellán, Silvia and Terenin, Alexander},
  journal={arXiv preprint arXiv:2605.05095},
  year={2026}
}
```

# References

[^spsr]: Holalkere, Sidhanth, David Bindel, Silvia Sellán, and Alexander Terenin. Stochastic Poisson Surface Reconstruction with One Solve using Geometric Gaussian Processes. International Conference on Machine Learning, 2025.

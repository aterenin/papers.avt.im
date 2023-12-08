+++
title = "Aligning Time Series on Incomparable Spaces"
[extra]
authors = [
    {name = "Samuel Cohen"},
    {name = "Giulia Luise", url = "https://giulslu.github.io"},
    {name = "Alexander Terenin", url = "https://avt.im/"},
    {name = "Brandon Amos", url = "http://bamos.github.io"},
    {name = "Marc Peter Deisenroth", url = "https://deisenroth.cc/"},
]
venue = {name = "AISTATS", date = 2020-08-26, url = "https://proceedings.mlr.press/v130/"}
buttons = [
    {name = "Paper", url = "https://proceedings.mlr.press/v130/cohen21a.html"},
    {name = "PDF", url = "https://arxiv.org/pdf/2006.12648"},
    {name = "Code", url = "https://github.com/samcohen16/Aligning-Time-Series"},
    {name = "Poster", url = "/presentations/2021-03-21-Aligning-Time-Series-Poster/2021-03-21-Aligning-Time-Series-Poster.pdf"},
    {name = "Video", url = "https://virtual.aistats.org/virtual/2021/poster/1628"},
]
katex = true
+++

Data is often gathered sequentially in the form of a time series, which consists of sequences of data points observed at successive time points. 
Dynamic time warping (DTW) defines a meaningful similarity measure between two time series.
Often times, the pairs of time series we are interested in are defined on different spaces: for instance, one might want to align a video with a corresponding audio wave, potentially sampled at different frequencies.
In this work, we propose *Gromov Dynamic Time Warping* (GDTW), a distance between time series on potentially incomparable spaces, and apply it to various settings, including barycentric averaging, generative modeling, and imitation learning.

# Time Series Alignment

Sakoe and Chiba[^dtw] consider the problem of aligning two time series `$\boldsymbol{x} \in \mathcal{X}^{T_x}$` and `$\boldsymbol{y} \in \mathcal{Y}^{T_y}$`, where potentially `$T_x \neq T_y$`.
This is formalized as

```
$$
\operatorname{DTW}(\boldsymbol{x},\boldsymbol{y}) = \min_{\mathbf{A} \in \mathcal{A}(T_x,T_y)} \langle\mathbf{D}, \mathbf{A}\rangle_{\operatorname{F}}
$$
```

where `$D_{ij} = d_\mathcal{X}(x_i,y_j)$` is the pairwise distance matrix and `$A_{ij}$` is  `$1$` if `$x_i$` and `$y_j$` are aligned, and `$0$` otherwise. 
A DTW alignment of two time series is shown in Figure 1.


{% figure(alt = ["Dynamic Time Warping"], src = ["align.svg"], dark_invert = [true]) %}
Figure 1: Alignment of two time series via Dynamic Time Warping.
{% end %}


A practical drawback of DTW is the need for both time series `$\boldsymbol{x}$` and `$\boldsymbol{y}$` to live on the same spaces, with a metric `$d_{\mathcal{X}}$`. 
This can cause the following issues.

* Alignment can fail for time series that are only close up to isometries, such as rotations and translations.
* The method doesn't apply to time series which are defined on different spaces, such as location coordinates for `$\boldsymbol{x}$` and pixel values for `$\boldsymbol{y}$`.

In such cases, defining a meaningful distance between samples from the two sequences is impractical as it would require detailed understanding of the objects we wish to study.

# Dealing with Incomparable Spaces

Motivated by connections between DTW and optimal transport,[^gw] we introduce a distance between time series `$\boldsymbol{x} \in \mathcal{X}^{T_x}$` and `$\boldsymbol{y} \in \mathcal{Y}^{T_y}$` defined on potentially incomparable metric spaces.
The key idea is to define a loss function which compares pairwise distances in `$\mathcal{X}^{T_x}$` with those in `$\mathcal{Y}^{T_y}$`. 
For this, we define the Gromov dynamic time warping distance as

```
$$
\operatorname{GDTW}(\boldsymbol{x},\boldsymbol{y})=\min_{\mathbf{A} \in \mathcal{A}(T_x,T_y)} \sum_{ijkl} \mathcal{L} \big(d_{\mathcal{X}}(x_i,x_k),d_{\mathcal{Y}}(y_j,y_l)\big) A_{ij}A_{kl},
$$
```

where `$d_{\mathcal{X}}$` is a distance defined on `$\mathcal{X}$`, and `$d_{\mathcal{Y}}$` a distance defined on `$\mathcal{Y}$`.
We solve the optimization problem over the set of alignment matrices by applying a Frank--Wolfe-inspired algorithm.
Results can be seen in Figure 2 for different rotations and translations of the original time series.


{% figure(src = ["align_gdtw.mp4"], dark_invert = [true]) %}
Figure 2: Alignment of two time series via Dynamic Time Warping and Gromov Dynamic Time Warping under different transformations.
{% end %}


Similarly to DTW, GDTW suffers from unpredictability when the time series is close to a change point of the optimal alignment matrix because of the discontinuity of derivatives. 
To remedy this, we introduce a softened version of this expression, mirroring the definition of soft DTW.[^sdtw]
This allows smoother derivatives when applying it to for instance generative modeling of time series and imitation learning.



# Applications

We showcase a number of applications of Gromov DTW.
We considered 3 settings: barycentric averaging, generative modeling, and imitation learning.


* **Barycentric averaging**: we extend the algorithm of Peyré et al.[^gw] to the sequential setting via coordinate descent on the GDTW objective. 
We plot barycenters under several warping approaches in Figure 3.
{% figure(alt = ["Barycenters of the Quickdraw dataset"], src = ["bary.svg"], dark_invert = [true]) %}
Figure 3: Barycenters of the Quickdraw dataset's fishes via various time warping approaches.
{% end %}


* **Generative modeling**: we extend the algorithm of Genevay et al.[^wgan] and Bunne et al.[^gwgan] to the sequential setting by leveraging GDTW as ground metric in an entropic Wasserstein objective. 
Samples can be observed in Figure 4.
{% figure(alt = ["Generated samples"], src = ["gan.svg"]) %}
Figure 4: Samples generated by the time series GAN trained on sequential MNIST.
{% end %}


* **Imitation learning**: We propose an approach to performing imitation learning when the agent and expert do not live on comparable spaces, which consists in minimizing the Gromov-DTW between expert demonstrations and agent rollouts. This is illustrated in Figure 5.
{% figure(alt=[false,"Expert trajectory and agent policy"], src = ["car.mp4","car_policy.svg"], dark_invert = [true,true]) %}
Figure 5: Expert trajectory (left, sequence of pixel images) and policy of an agent (right, sequence of points in `$\mathbb{R}^2$`) learned by imitation learning.
{% end %}


# Summary

We introduce a distance between time series living on potentially incomparable spaces, Gromov DTW, which significantly broadens the range of applications of previous time-series metrics like DTW. 
We also propose a smoothed version that alleviates the discontinuity of GDTW's gradient. 
Gromov DTW is a general concept for comparing two time series and can be applied to a number of applications, including barycentric averaging, generative modeling and imitation learning.

# References

[^dtw]: H. Sakoe, S. Chiba. Dynamic Programming Algorithm Optimization for Spoken Word Recognition. ICASSP, 2018.

[^gw]: G. Peyré, M. Cuturi, J. Solomon. Gromov--Wasserstein Averaging of Kernel and Distance Matrices. ICML, 2016.

[^sdtw]: M. Cuturi, M. Blondel. Soft-DTW: A Differentiable Loss Function for Time-Series. ICML, 2017.

[^wgan]: A. Genevay, G. Peyré, M. Cuturi. Learning Generative Models with Sinkhorn Divergences. AISTATS, 2018.

[^gwgan]: C. Bunne, D. Alvarez-Melis, A. Krause, S. Jegelka. Learning Generative Models Across Incomparable Spaces. ICML, 2019.

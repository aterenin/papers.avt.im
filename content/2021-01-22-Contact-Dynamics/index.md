+++
title = "Learning Contact Dynamics using Physically Structured Neural Networks"
[extra]
authors = [
    {name = "Andreas Hochlehnert", url = ""},
    {name = "Alexander Terenin", url = "https://avt.im/"},
    {name = "Steindór Sæmundsson", url = ""},
    {name = "Marc Peter Deisenroth", url = "https://deisenroth.cc/"},
]
venue = {name = "AISTATS", date = 2021-04-13, url = "https://aistats.org/aistats2021/"}
buttons = [
    {name = "Paper", url = "http://proceedings.mlr.press/v130/hochlehnert21a.html"},
    {name = "PDF", url = "https://arxiv.org/pdf/2102.11206"},
    {name = "Code", url = "https://github.com/libeanim/contact-symplectic-integrator-network"},
    {name = "Poster", url = "/presentations/2021-03-21-Contact-Dynamics-Poster/2021-03-21-Contact-Dynamics-Poster.pdf"},
]
video = "https://virtual.aistats.org/virtual/2021/poster/1764"
katex = true
+++

Learning models of physical systems can sometimes be difficult. 
Vanilla neural networks---like residual networks---particularly struggle to learn invariant properties like the conservation of energy which is fundamental to physical systems.
To counteract this, a number of recent works such as *Hamiltonian Neural Networks* and *Variational Integrator Networks* introduce *inductive biases*, also referred to as *physics priors*, which improve reliability of predictions and speed up learning.
These network classes exhibit good approximation behavior for continuous physical systems but they are fundamentally limited to smooth dynamics, and not designed to handle non-smooth physical behavior, such as resolving collision events between different objects.
Such behavior is of key interest in robotics, and other areas of engineering.
In this work, we explore neural network architectures designed for accurately modeling contact dynamics, which incorporate the structure necessary to reliably resolve non-smooth collision events.

# Contact Dynamics 

*Contact dynamics* are a class of equations which describe the motion of physical systems consisting of multiple solid objects which interact with each other and their environment.
One of these equations' defining features is that when two objects collide, their velocities change direction *instantaneously* in a non-smooth manner---this describes, for instance, how a bouncing ball immediately changes direction upon hitting the ground, resulting from a transfer of momentum and other physical considerations.


{% figure(src = ["bb_resnet_phase.mp4"], dark_invert = [true]) %}
**Illustration:** here, we see a residual network, trained to predict the trajectory of a bouncing ball, struggle to resolve contact events. Since contacts trigger a discontinuous jump in velocity space, they are difficult to model in a black-box fashion, causing the residual network to incorrectly approximate them through spurious smooth dynamics. We explore the use of physical inductive biases to alleviate these issues.
{% end %}


Because of the resulting non-smoothness and non-linearity, contacts dynamics are considered notoriously difficult to compute. 
For example, a numerical regime must decide whether to enforce non-interpenetration constraints by precisely calculating contact times using an optimization procedure, or instead allow physically incorrect interpenetration.
Below, we illustrate sample numerical trajectory of a bouncing ball.


{% figure(alt = ["Contact time","Contact state"], src = ["contact-time.svg","contact-state.svg"], dark_invert = [true,true]) %}
**Example:** integration scheme for a bouncing ball that enforces constraints exactly. Initially, the ball is time-stepped until a contact with the floor is detected through interpenetration at time `$t_1$`. Then, the trajectory is (a) linearly interpolated to find the *contact time* `$t_c$` where contact occurs between the ball and floor. Finally, the contact state at time `$t_c$` is calculated, a transfer of momentum between `$t_c^-$` and `$t_c^+$` is performed, and the ball is time-stepped as usual to time `$t_1$`.
{% end %}


These difficulties are further compounded when the equations of motion of the system under study are unknown, which occurs in robotics when learning to interact with unknown objects.
The aim of this work is to explore neural network architectures which are capable of accurately modeling such systems.

# Central Difference Lagrange Networks

We begin with the perspective of *neural ordinary differential equations*, which view deep networks as discretizations of continuous-time dynamical systems.
Our system's state is defined as velocity pairs `$(\mathbf{Q}, \mathbf{\dot Q})$`. In-between contact events, the trajectories follow the *Euler-Lagrange equations*

```
$$
\frac{\partial L}{\partial \mathbf{Q}} - \frac{\mathrm{d}}{\mathrm{d}t} \frac{\partial L}{\partial \mathbf{\dot Q}} = 0
$$
```

where `$L=T-V$` is the Lagrangian with the potential `$V$` and kinetic energy `$T$`.
At contact times, the above equations do not hold, and a set of instantaneous *transfer of momentum* equations apply instead.

To model these dynamics, we adopt an approach similar in spirit to *Variational Integrator Networks*[^vins] by modeling `$V$` using a fully connected neural network and discretizing the resulting equations of motion to construct a recurrent network architecture.
The choice of differential equation class and discretization scheme determines the inductive bias that is introduced. 
These biases can include physical properties such as conservation of momentum and of energy, as well as other fundamental mechanical characteristics.

To design a scheme for contact dynamics, we employ the Central Difference--Lagrange (CDL) scheme,[^cdl] whose equations form the basis of our network architecture.
Between contact times, the dynamics evolve smoothly, which we denoted by `$(\cdot)^S$`, in a manner that mirrors variational integrator networks.
During contact events, these equations are augmented by a contact term, denoted by `$(\cdot)^{C}$`, that handles the transfer of momentum and makes sure that (1) Newton's restitution law, as well as (2) the law of conservation of momentum both hold.
Below, we illustrate how the different states in the CDL-Network are calculated.


{% figure(alt = ["Contact network"], src = ["contact-network.svg"], dark_invert=[true]) %}
**Illustration:** a CD-Lagrange network. 
Here, we begin from initial states `$(\mathbf{Q}_0,\mathbf{\dot Q}_{\frac{1}{2}})$`. 
We calculate the next position `$\mathbf{Q}_1$`, and proceed to calculate the next velocity `$\mathbf{\dot Q}_{1 + \frac{1}{2}} = \mathbf{\dot Q}_{1 + \frac{1}{2}}^S + \mathbf{\dot Q}_{1 + \frac{1}{2}}^C$` as a sum of smooth and contact terms.
These terms are in turn calculated using the conservative forces `$\mathbf{F}$` and the impulse `$\mathbf{I}$`, which are calculated from the parameterized Lagrangian, whose potential energy is given by a fully connected network.
{% end %}


# Touch Feedback

In an unknown system, one must determine when a system's trajectory evolves according to smooth dynamics, and when it evolves according to contact events.
Our results suggest that external touch feedback---such as that available from an idealized touch sensor---is necessary to make the problem tractable. 
This is handled by introducing a *contact network* `$\hat{c}$` which learns to predict contact events, using training data obtained via said touch sensor.
Without this, the network struggles to differentiate noise from real contact events, as small contact events and noise both generate similar data.
We illustrate this behavior on the following examples.

* **Bouncing ball**. Here, we see that without touch feedback, the CD--Lagrange network struggles to learn the contact events properly and instead approximates the discontinuous behavior using the potential network.
With training data that includes touch feedback, we see that performance, shown below, is significantly better.
{% figure(src = ["bb_cdln_phase.mp4"], dark_invert=[true]) %}
**(a)** Bouncing ball: CD--Lagrange
{% end %}


* **Newton's cradle**. As an additional baseline, we employ a vanilla residual network (ResNet) as well as a residual network with additional contact inputs (ResNet contact).
The CD-Lagrange network, shown below, exhibits the best approximation behavior and learn both the potential and contact events more accurately than the residual networks.
{{ figure(src = ["nc_cdln.mp4","nc_resnet.mp4"], subcaption = ["**(a)** Newton's cradle: CD--Lagrange","**(b)** Newton's cradle: ResNet"], dark_invert=[true,true]) }}


# Summary

State-of-the-art physics-inspired neural networks generally struggle to learn contact dynamics.
Central-Difference-Lagrange networks are a class of networks that not only exhibit strong conservation properties, comparable to other physically structured neural networks, but also allow accurate learning of contact dynamics from observed data.
In this regime, the information available to the network when making predictions has a significant effect on performance: the addition of touch feedback sensor data ensures that noise and contact events are correctly differentiated.
We hope these contributions enable neural network models to be used in wider settings.

# References

[^vins]: S. Sæmundsson, A. Terenin, K. Hofmann, M. P. Deisenroth. Variational Integrator Networks for Physically Structured Embeddings. AISTATS, 2020.

[^cdl]: F.-E. Fekak, M. Brun, A. Gravouil, and B. Depale. A new heterogeneous asynchronous explicit--implicit time integrator for nonsmooth dynamics. Computational Mechanics, 60(1):1--21, 2017.
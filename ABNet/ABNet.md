# ABNet: Attention BarrierNet for Safe and Scalable Robot Learning

Wei Xiao, Tsun-Hsuan Wang, and Daniela Rus Computer Science and Artificial Intelligence Lab Massachusetts Institute of Technology Cambridge, MA 02139 Corresponding: weixy@mit.edu 

## Abstract

Safe learning is central to AI-enabled robots where a single failure may lead to catastrophic results. Barrier-based method is one of the dominant approaches for safe robot learning. However, this method is not scalable, hard to train, and tends to generate unstable signals under noisy inputs that are challenging to be deployed for robots. To address these challenges, we propose a novel Attention BarrierNet (ABNet) that is scalable to build larger foundational safe models in an incremental manner. Each head of BarrierNet in the ABNet could learn safe robot control policies from different features and focus on specific part of the observation. In this way, we do not need to one-shotly construct a large model for complex tasks, which significantly facilitates the training of the model while ensuring its stable output. Most importantly, we can still formally prove the safety guarantees of the ABNet. We demonstrate the strength of ABNet in 2D robot obstacle avoidance, safe robot manipulation, and vision-based end-to-end autonomous driving, with results showing much better robustness and guarantees over existing models. <sup>1</sup> 

## 1 Introduction

A Safety-Guaranteed Learning System with Attention Mechanism 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-01/69559fc6-5d48-441b-9cdf-fe903d11b9dc/8af6998a0fb465a05c8149e71a5997dd0ca00ab37bf2def38dec2308e5c5a67d.jpg)



Figure 1: The proposed ABNet that is robust, scalable and generates stable output while guaranteeing safety for robots. Each head of BarrierNet in the model could learn safe control policies with attention on different observation feature in a scalable or one-shot manner.


Robot learning usually requires to leverage scalable training and vast amount of data. There are many large models for complex robotic tasks including manipulation, locomotion, autonomous driving [7] [32] [35]. However, these models are not trustworthy and have no safety guarantees. Existing methods that incorporate guarantees or certificates into neural networks are not scalable and hard to train [23] [39] [36]. It is desirable to merge these safe models for complicated robot tasks. Traditional mixture of expert methods [31] [27] [42] or other merging approaches [14] [26] [34] are hard to retain the safety of the models. In this work, we explore to leverage the collective power of many safety-critical models to handle complex tasks while preserving the safety of the merged models. 

There are various definitions of safety for robotics and autonomy, and safety can be basically defined as something bad never happens. Mathematically, safety can be defined as a continuously differentiable constraint with respect to the system state and it can be further captured by the forward invariance of the safe set over such a constraint [1] [38] [13]. In other words, we can use different constraints and approaches to enforce safety. The way we learn such safety enforcement methods may depend on the focused observation feature, which corresponds to the attention mechanism. For instance, some human drivers may focus on the left lane boundary in driving in order to achieve safe lane keeping, while others may focus on the right lane boundary, as shown in Fig. 1. Both attention mechanisms can achieve similar purpose. Merging these models or attention mechanisms enables us to build robust and powerful learning models. However, retaining safety is non-trivial. 

In the literature, barrier-based learning methods [28] [23] [33] [40], such as the BarrierNet [39] [36] [20], are widely used to equip deep learning systems with safety guarantees. We may incorporate control-theoretic based optimizations into learning systems in the form of differentiable quadratic programs (dQPs) [3]. There are several limitations of these barrier-based learning methods: (i) it can only implement a single safety enforcement method as the last layer of the neural network, which is not scalable to larger safe learning models; (ii) the model is not robust such that it is hard to be trained to work for complicated robotic applications; (iii) these methods tend to generate unstable output under noisy observation, which is intractable to be deployed for robots. 

In this paper, we propose a novel Attention BarrierNet (ABNet) to merge many safety-critical models while preserving the safety guarantees.The ABNet is scalable, robust to noise, and easy to be trained in an incremental manner. As shown in Fig. 1, we may build multi-head BarrierNets within the ABNet. Each head of the BarrierNet may pay attention to different observation features to generate a safe control policy. We linearly combine the outputs of all the BarrierNets in a way that is provably safe. The weights of this combination quantify the importance of each head of BarrierNet, and they are trainable. The structure of the ABNet allows us to build larger foundational safe models for various and complicated robotic applications as we can incrementally train safe models corresponding to different robot skills and this will simply increase the head h of BarrierNets. 

In summary, we make the following new contributions: 

• We propose a novel ABNet that merges many safety-critical learning models, and this new model is scalable, robust, and easy to be trained. 

• We formally prove the safety guarantees of the proposed ABNet. 

• We demonstrate the strength and effectiveness of our model on a variety of robot control tasks, including 2D robot obstacle avoidance, safe robot manipulation, and vision-based end-to-end autonomous driving in an open dataset. We also show that existing models/policies merging could make safety worse in complicated tasks (such as in vision-based driving). 

## 2 Preliminaries and Problem Formulation

In this section, we present background on the forward invariance with High-Order Control Barrier Functions (HOCBFs) that is widely used to enforce safety, as well as introduce the BarrierNet. 

Forward Invariance with HOCBFs. Consider an affine control system defined as: 

$$
\dot {\boldsymbol {x}} = f (\boldsymbol {x}) + g (\boldsymbol {x}) \boldsymbol {u}\tag{1}
$$

where $\pmb { x } \in \mathbb { R } ^ { n }$ is the system state, $f : \mathbb { R } ^ { n } \to \mathbb { R } ^ { n }$ and $g : \mathbb { R } ^ { n }  \mathbb { R } ^ { n \times q }$ are locally Lipschitz, and $\pmb { u } \in U \subset \mathbb { R } ^ { q }$ , where U denotes a control constraint set. ˙x denotes the time derivative of state x. 

We begin with some definitions before introducing the HOCBF. 

Definition 2.1. (Forward invariance $I I J .$ A set $C \subset \mathbb { R } ^ { n }$ is forward invariant for system (1) if its solutions for some ${ \pmb u } \in U$ starting at any $\begin{array} { r } { \pmb { x } ( 0 ) \in C \ s a t i s f y \ \pmb { x } ( t ) \in C , \forall t \geq 0 . } \end{array}$ 

Definition 2.2. (Class K function [16]): A Lipschitz continuous function $\alpha : [ 0 , a ) \to [ 0 , \infty ) , a > 0$ belongs to class K if it is strictly increasing and $\alpha ( 0 ) = 0$ 

Definition 2.3. (Relative degree [16]): The relative degree of a (sufficiently many times) differentiable function $b : \mathbb { R } ^ { n } \to \mathbb { R }$ with respect to system (1) is defined as the number of times that we need to differentiate b along the system (1) until any component of the control u explicitly shows up in the corresponding derivative. 

Since a function $b ( { \pmb x } )$ is used to defined a safety constraint $b ( { \pmb x } ) \geq 0 .$ , we refer to the relative degree of the constraint as the relative degree of the function. Consider a safety constraint $b ( { \pmb x } ) \geq 0$ with relative degree m for system (1), where $b : \mathbb { R } ^ { n } $ <sup>R</sup> is continuously differentiable, we recursively define a sequence of CBFs $\psi _ { i } : \mathbb { R } ^ { n }  \mathbb { R } , i \in \{ 1 , \ldots , m \}$ in the form: 

$$
\psi_ {i} (\boldsymbol {x}) := \dot {\psi} _ {i - 1} (\boldsymbol {x}) + \alpha_ {i} \left(\psi_ {i - 1} (\boldsymbol {x})\right), i \in \{1, \dots , m \},\tag{2}
$$

where $\psi _ { 0 } ( { \pmb x } ) : = b ( { \pmb x } )$ , and $\alpha _ { i } , i \in \{ 1 , \ldots , m \}$ are class $\mathcal { K }$ functions. 

We further define a sequence of safe sets $C _ { i } , i \in \{ 1 \ldots , m \}$ corresponding to (2) in the form: 

$$
C _ {i} := \left\{\boldsymbol {x} \in \mathbb {R} ^ {n}: \psi_ {i - 1} (\boldsymbol {x}) \geq 0 \right\}, i \in \{1, \dots , m \}.\tag{3}
$$

Definition 2.4. (High Order Control Barrier Function (HOCBF) [38]): Let $C _ { i } , i \in \{ 1 , . . . , m \}$ and $\psi _ { i } , i \in \{ 1 , \ldots , m \}$ be defined by (3) and (2), respectively. A function $b : \mathbb { R } ^ { n } \to \mathbb { R }$ is a HOCBF if there exist class K functions $\alpha _ { i } , i \in \{ 1 \ldots , m \}$ such that 

$$
\sup _ {\boldsymbol {u} \in U} [ L _ {f} \psi_ {m - 1} (\boldsymbol {x}) + [ L _ {g} \psi_ {m - 1} (\boldsymbol {x}) ] \boldsymbol {u} + \alpha_ {m} (\psi_ {m - 1} (\boldsymbol {x})) ] \geq 0,\tag{4}
$$

for all $\pmb { x } \in \cap _ { i = 1 } ^ { m } C _ { i } . \ L _ { f }$ and $L _ { g }$ denote Lie derivatives w.r.t. x along f and g, respectively. 

Theorem 2.5 ([38]). Given a HOCBF b(x) from Def. 2.4, $i f { \mathbf { \boldsymbol { x } } } ( 0 ) \in \cap _ { i = 1 } ^ { m } C _ { i } ,$ , then any Lipschitz continuous controller $\pmb u(t)$ that satisfies the constraint in $( 4 ) , \forall t \geq 0$ renders $\cap _ { i = 1 } ^ { m } C _ { i }$ forward invariant for system (1).

The HOCBF is a general form of the CBF [1] [13], i.e., setting the relative degree $m = 1$ of a safety constraint $b ( { \pmb x } ) \geq 0$ will reduce a HOCBF to a CBF. CBFs/HOCBFs are widely used to transform nonlinear optimal control problems into a sequence of Quadratic Programs (QPs) that are very efficient to solve while preserving the safety guarantees of the system. 

BarrierNet. The BarrierNet [39] is a neural network layer that incorporates CBF/HOCBF-based QPs as differentiable QPs (dQPs) [3], in which all the CBFs/HOCBFs are differentiable in terms of their parameters (such as those in class $\mathcal { H }$ functions). Those parameters are crucial to the system conservativeness or performance in guaranteeing safety. In summary, the BarrierNet frees us from handing-tuning all the parameters in safety-critical controls, and simply uses data to optimize them. Referring to Fig. 1, a BarrierNet only has a single head in the model $( \mathrm { i . e . , } h = 1 )$ and it is placed as the last layer of the model when used in conjunction with other neural networks (such as CNN and LSTM). 

In this paper, we consider the following safe learning problem: 

Problem. Given (a) a system with dynamics in the form of (1); (b) a state-feedback nominal controller $\pmb { \pi } ^ { * } ( \pmb { x } ) = \pmb { u } ^ { * }$ (such as a model predictive controller) that provides the training label; (c) a set of safety constraints $b _ { j } ( \pmb { x } ) \geq 0 , j \in S \ ( b _ { j }$ is continuously differentiable, S is a constraint set); (d) a neural network controller $\pi ( x , z | \theta ) = \mathbf { \bar { u } }$ parameterized by θ (under observation z); 

Our goal is to find the optimal parameter 

$$
\theta^ {*} = \arg \min _ {\theta} \mathbb {E} _ {\boldsymbol {x}, \boldsymbol {z}} [ \ell (\pi^ {*} (\boldsymbol {x}), \pi (\boldsymbol {x}, \boldsymbol {z} | \theta)) ],\tag{5}
$$

while satisfying all the safety constraints in (c) and the dynamics constraint (a). <sup>E</sup> is the expectation, and ℓ is a loss function. 

## 3 Attention BarrierNet

In this section, we present the architecture of the Attention BarrierNet (ABNet) and formally prove its safety guarantees in learning systems. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-01/69559fc6-5d48-441b-9cdf-fe903d11b9dc/e6178a9a1a187550683bb82bf1c77e89813f078e471b4a868285ea9f2575e384.jpg)



Figure 2: Architecture of multi-head BarrierNets (i.e., ABNet). The ABNet is usually used in conjunction with any other neural networks and can be implemented in parallel. The parameters (inputs) of each head of BarrierNet are the outputs of previous layers (such as CNN or LSTM).


## 3.1 Multi-head BarrierNets

We can use a BarrierNet to transform the constrained optimal control in the considered problem into the following differentiable QP, which forms a head of BarrierNet in the model: 

$$
\boldsymbol {u} _ {k} = \arg \min _ {\boldsymbol {u} (t) \in U} \frac {1}{2} \boldsymbol {u} (t) ^ {T} H (\boldsymbol {z} _ {k} | \boldsymbol {\theta} _ {h, k}) \boldsymbol {u} (t) + F ^ {T} (\boldsymbol {z} _ {k} | \boldsymbol {\theta} _ {f, k}) \boldsymbol {u} (t)\tag{6}
$$

s.t. 

$$
\begin{array}{c} L _ {f} \psi_ {j, m - 1} (\boldsymbol {x}, \boldsymbol {z} | \theta_ {p}) + [ L _ {g} \psi_ {j, m - 1} (\boldsymbol {x}, \boldsymbol {z} | \theta_ {p}) ] \boldsymbol {u} + p _ {m, k} (\boldsymbol {z} _ {k} | \theta_ {p, k} ^ {m}) \alpha_ {j, m} (\psi_ {j, m - 1} (\boldsymbol {x}, \boldsymbol {z} | \theta_ {p})) \geq 0, j \in S, \\ \psi_ {j, i} (\boldsymbol {x}, \boldsymbol {z} | \theta_ {p}) = \dot {\psi} _ {j, i - 1} (\boldsymbol {x}, \boldsymbol {z} | \theta_ {p}) + p _ {i} (\boldsymbol {z} | \theta_ {p} ^ {i}) \alpha_ {j, i} (\psi_ {j, i - 1} (\boldsymbol {x}, \boldsymbol {z} | \theta_ {p})), i \in \{1, \ldots , m - 1 \}, j \in S, \\ \psi_ {j, 0} (\boldsymbol {x}, \boldsymbol {z} | \theta_ {p}) = b _ {j} (\boldsymbol {x}), j \in S, \qquad t = \omega \Delta t + t _ {0}, \omega \in \{0, 1, \ldots \}, \end{array}\tag{7}
$$

where $k \in \{ 1 , \ldots , h \}$ , and h is the number of heads of BarrierNet (as shown in Fig. 1). $p _ { i } \geq 0 , i \in$ $\{ 1 , \ldots , m - 1 \} , p _ { m , k } \ge 0$ are penalty functions on the class $\mathcal { H }$ functions $\alpha _ { j , i } , i \in \{ 1 , \ldots , m \} , j \in S$ that address the conservativeness of the model (e.g., how far away the system state should stay form the unsafe set bound in order to maintain safety). All the HOCBFs corresponding to the safety constraints share the same penalty functions, but they may use different ones in which case $p _ { i }$ and $p _ { m , k }$ will be dependent on $j , j \in S$ . The derivatives of the observation z in the above are omitted, as shown in [39]. $H ( z _ { k } | \theta _ { h , k } ) \stackrel { } { \in } \mathbb { R } ^ { q \times q }$ is positive definite, and $H ^ { - 1 } ( z _ { k } | \theta _ { h , k } ) F ( z _ { k } | \theta _ { f , k } )$ can be interpreted as a reference control (the output of previous network layers). $\theta : = ( \theta _ { h } , k , \tilde { \theta _ { f , k } } , \theta _ { p , k } ^ { m } , \theta _ { p } ) , k \in \{ 1 , \dots , h \}$ where $\theta _ { p } : = ( \theta _ { p } ^ { 1 } , \ldots , \theta _ { p } ^ { m - 1 } )$ are all trainable parameters of the neural network. $z _ { k }$ is the observation of the BarrierNet head $k , k \in \{ 1 , \ldots , h \}$ , and it is possible that all heads share the same observation, i.e. $z _ { k } = z , \forall k \in \{ 1 , \ldots , h \}$ $\Delta t > 0$ is the discretized time interval, and $t _ { 0 }$ is the initial time. 

Attention mechanism. Each head of BarrierNet may learn safe self-attention even if all the Barrier-Nets have the same observation z. The parameter $p _ { m , k } ^ { m }$ may be learned from different input features via random initialization, and it determines the conservativeness of the model in guaranteeing safety. On the other hand, we may also make each head of BarrierNet focus on different observations $z _ { k }$ The observation $z _ { k }$ may come from different parts of the sensor observation (such as the left lane boundary and right lane boundary in driving shown in Fig. 1), or even different perceptions (such as vision, lidar, etc.) 

Cross connection. It can be noted from (7) that each head of BarrierNet $k \in \{ 1 , \ldots , h \}$ has some cross connection with other heads, as also shown in $\mathrm { F i g }$ . 1. In other words, $\psi _ { j , i } ( \pmb { x } , \pmb { z } | \theta _ { p } ) , i \in$ $\{ 1 , \ldots , m - 1 \} , j \in S$ are formulated in the same way through the shared parameter $\theta _ { p }$ (independent from k). This structure is to ensure (i) the construction for provable safety (as shown later), and (ii) some shared information across different heads of BarrierNet as they all generate safe control policies for the same system (1). 

Fusion. Another important consideration is how should we fuse all these controls ${ \pmb u } _ { k } , k \in \{ 1 , . . . , h \}$ while preserving the safety property of each head of the BarrierNet. We propose the following form: 

$$
\boldsymbol {u} = \sum_ {k = 1} ^ {h} w _ {k} \boldsymbol {u} _ {k}, \quad \text { where } \sum_ {k = 1} ^ {h} w _ {k} = 1.\tag{8}
$$

Algorithm 1 Construction and training of ABNet

Input: the problem setup (a)-(d) given in the problem formulation (end of Sec. 2).
Output: a robust and safe controller u for system (1).
(a) Formulate each head of BarrierNet as in (6) s.t. (7).
(b) Build the cross connection among BarrierNets via $p_{i}(z|\theta_{p}^{i}), i \in \{1, \ldots, m-1\}$ .
(c) Fuse all the heads of BarrierNet as in (8).
if Scalable training then
    Decouple $p_{i}(z|\theta_{p}^{i}), i \in \{1, \ldots, m-1\}$ and define them for each BarrierNet.
    Train each head of BarrierNet, respectively.
    Choose a $p_{i}(z|\theta_{p}^{i}), i \in \{1, \ldots, m-1\}$ from one of the BarrierNets to build cross connection.
    Fuse all the BarrierNets via (9).
else
    Directly train the ABNet via reverse mode error back propagation.
end if 

In the above, $w _ { k } \geq 0 , k \in \{ 1 , \ldots , h \}$ are trainable parameters. The composition of all the heads of BarrierNet (6) s.t., (7) in the form of (8) is our proposed ABNet, as shown in Fig. 2. The safety guarantees of the ABNet is shown in the following theorem: 

Theorem 3.1. (Safety of ABNets) Given the multi-head BarrierNets formulated as in (6) s.t. (7). If the system (1) is initially safe $( i . e . , b _ { j } ( { \pmb x } ( t _ { 0 } ) ) \geq 0 , \forall j \in S )$ , then a control policy u from the ABNet output (8) guarantees the safety of system (1), i.e., $b _ { j } ( { \pmb x } ( t ) ) \geq 0 , \forall j \in S , \forall t \geq t _ { 0 } .$ 

All the proofs for theorems are given in Appendix A. If the system is not initially safe $( \mathrm { i . e . , } b _ { j } ( { \pmb x } ( t _ { 0 } ) ) <$ $0 , \exists j \in S )$ , then the system state x of (1) will be driven to the safe side of the state space due to the Lyapunov property of CBF/HOCBFs [1] [38]. This enables the possibility of utilizing data that violates safety to conduct adversary training of the ABNet. 

Natural noise filter. The ABNet is a natural noise filter since $w _ { k } \in [ 0 , 1 ] , \forall k \in \{ 1 , \dots , h \}$ in (8). This can ensure that the output u of the model is stable with a large enough head number h if all the BarrierNets have different observation $z _ { k }$ for the current environment. This feature makes ABNet a very robust controller for robotic systems, and thus, ABNet can generate smooth signals. 

Theorem 3.2. (Safety of merging of ABNets) Given two ABNets with each formulated as in (8) and (6) s.t. (7), the merged model using the form as in (8) again guarantees the safety of system (1). 

## 3.2 Model Training

The ABNet can be trained incrementally or in one-shot. This is due to the fact that each head of BarrierNet can generate a control policy that is applicable to system (1). The linear combination weights $w _ { k } , k \in \left\{ 1 , \ldots , h \right\}$ in the ABNet denote the importance of the corresponding control policies. 

Scalable training. In ABNet, we may train each head $k , k \in \{ 1 , \ldots , h \}$ of the BarrierNet in a scalable way as we wish to minimize the loss between their output $\mathbf { \Delta } \mathbf { u } _ { k }$ and the label $\pmb { u } ^ { * }$ as well. The training can be done using the batch QP training method proposed in [3]. There are some cross connections via $p _ { i } ( z | \theta _ { p } )$ between BarrierNets in the ABNet that may prevent the implementation of the training. We may address this by training a $p _ { i } ( z | \theta _ { p } )$ for each head of the BarrierNet. After we train all heads of the BarrierNet, we may fix the parameters of those models, choose a $p _ { i } ( z | \theta _ { p } )$ from one of the BarrierNets (or take an average of all $p _ { i } ( z | \theta _ { p } )$ among the BarrierNets) to build the cross connection, and train the weights w for some more iterations. Another way is to fuse these BarrierNets by their testing loss. In other words, the weight $w _ { k } , k \in \{ 1 , \ldots , h \}$ can be determined by: 

$$
w _ {k} = \frac {\frac {1}{\ell_ {k} (\boldsymbol {u} _ {k} , \boldsymbol {u} ^ {*})}}{\sum_ {k = 1} ^ {h} \frac {1}{\ell_ {k} (\boldsymbol {u} _ {k} , \boldsymbol {u} ^ {*})}},\tag{9}
$$

where $\ell _ { k }$ is a loss function. We may also use some exponential functions of the losses to determine w<sub>k</sub> similarly as in the above equation. 

If we already have some trained ABNet, and we wish to add some new capabilities (such as safe driving by only focusing on the left lane boundary) to the model, then we can train some heads of BarrierNets based on the new data we have. Finally, we can fuse the models similarly with safety guarantees as shown in Thm. 3.2. This shows the scalability of the proposed ABNet that allows us to build larger foundational safe models in an incremental way. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-01/69559fc6-5d48-441b-9cdf-fe903d11b9dc/ebfed73ea9b18d667d8f023dfac853cae3229104377571461a5fd380b0e96edf.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-01/69559fc6-5d48-441b-9cdf-fe903d11b9dc/6b23976d8db1b7a8d61329701e65cf90ccfbf149434ef7903d77513b4094b2a2.jpg)



log(h): log number of BarrierNet heads with scalable training



Figure 3: 2D robot obstacle avoidance closed-loop testing control profiles (left) and ABNet performance with the increasing of BarrierNet heads using scalable training (right). This scalable training for ABNet is with safety guarantees. The controls are subject to input noise, and thus are non-smooth.


One-shot training. The one-shot training of the ABNet can be directly done using the traditional reverse mode automatic differentiation. In addition to the loss between the eventual output u of the ABNet and the label u<sup>∗</sup>, we may also consider the losses on ${ \pmb u } _ { k } , k \in \{ 1 , \ldots , h \}$ , as well as on the reference controls $H ^ { - 1 } ( z _ { k } | \theta _ { h , k } ) F ( z _ { k } | \theta _ { f , k } )$ , in order to improve the training performance. 

The construction and training of the ABNet involve the formulation of each head of BarrierNet as in (6) s.t. (7), the BarrierNet fusion as in (8), and the scalable or one-shot training as shown above. We summarize this process in Alg. 1. 

## 4 Experiments


Table 1: 2D robot obstacle avoidance closed-loop testing under noisy input and comparisons with benchmarks.


<table><tr><td>MODEL</td><td>MSE(↓)</td><td>SAFETY (≥0)</td><td>CONSER. (≥0&amp;↓)</td><td><eq>u_1</eq> UNCER-TAINTY (↓)</td><td><eq>u_2</eq> UNCER-TAINTY (↓)</td><td>THEORET. GUAR.</td></tr><tr><td>E2E [18]</td><td>0.007±0.004</td><td>-14.140</td><td>-2.976±3.770</td><td>0.063</td><td>0.049</td><td>×</td></tr><tr><td>E2Es-MCD [12]</td><td>0.004±0.001</td><td>-2.087</td><td>-1.341±0.824</td><td>0.041</td><td>0.026</td><td>×</td></tr><tr><td>E2Es-DR [17]</td><td>0.080±0.006</td><td>-35.130</td><td>-3.176±4.299</td><td>0.032</td><td>0.020</td><td>×</td></tr><tr><td>DFB [23]</td><td>0.013±0.003</td><td>36.659</td><td>47.810±4.377</td><td>0.062</td><td>0.052</td><td>√</td></tr><tr><td>BNET [39]</td><td>0.014±0.006</td><td>5.045</td><td>7.966±1.287</td><td>0.074</td><td>0.047</td><td>√</td></tr><tr><td>BNET-UP [36]</td><td>0.008±0.004</td><td>5.988</td><td>8.573±1.738</td><td>0.054</td><td>0.028</td><td>×</td></tr><tr><td>ABNET-10-SC (OURS)</td><td>0.011±0.007</td><td>5.731</td><td>6.269±0.319</td><td>0.065</td><td>0.027</td><td>√</td></tr><tr><td>ABNET-10 (OURS)</td><td>0.008±0.005</td><td>12.639</td><td>13.887±1.323</td><td>0.049</td><td>0.030</td><td>√</td></tr><tr><td>ABNET-100 (OURS)</td><td>0.012±0.006</td><td>10.122</td><td>11.729±0.816</td><td>0.049</td><td>0.013</td><td>√</td></tr></table>


In this section, we conduct several experiments to answer the following questions: 


• Does our method match the theoretic results in experiments and guarantee the safety of robots in various tasks quantitatively, qualitatively and is it scalable? 

• How does our method compare with state-of-the-art models (baseline E2E, safety-guaranteed models, policies merging, models merging) in enforcing safety constraints? 

• The benefit of models/policies merging and the robustness of our models in safety and smoothness? 

Benchmark models: We compare with (i) baseline: Tables 1, 2–single end-to-end learning model (E2E) [18] and Table 3–single vanilla end-to-end (V-E2E) model [2], (ii) safety guaranteed models: single BarrierNet (BNet) [39], Deep forward and backward (DFB) model [23], (iii) policies merging: BarrierNet policies merged with uncertainty propagation (BNet-UP) [36] that employs Gaussian kernels with Scott’s rule [30] to select the bandwidth, (iv) models merging: E2Es merged with Monte-Carlo Dropout (E2Es-MCD) [12], E2Es merged with Deep Resembles (E2Es-DR) [17]. 

Our models: Sec. 4.1 and 4.2: ABNet trained in a scalable way with 10 heads (ABNET-10-SC), ABNet trained in one shot with 10 heads (ABNET-10), ABNet trained in one shot with 100 heads (ABNET-100). Sec. 4.3: our ABNet trained in one shot with 10 heads using the same input images (ABNET), ABNet with attention images and 10 heads (ABNET-ATT), our ABNet first trained with ABNET scaled/augmented by ABNET-ATT (20 heads, ABNET-SC). 

Evaluation metrics: The evaluation metrics in all the tables are defined as follows: mean square error of the model testing (MSE), satisfaction of safety constraints where non-negative values mean safety guarantees (SAFETY), system conservativeness (CONSER.), steering control u uncertainty (u UNCERTAINTY), acceleration control u<sub>2</sub> uncertainty $( u _ { 2 }$ UNCERTAINTY), and theoretical safety guarantees (THEORET. GUAR.) respectively. All the metrics are explicitly defined in Appendx B. 

## 4.1 2D Robot Obstacle Avoidance

We aim to find a neural network controller for a 2D robot that can drive the robot from an initial location to an arbitrary destination while avoiding crash onto the obstacle. All the models (h copies/heads) have the same input (with uniformly distributed noise, 10% of the input magnitude in testing). The detailed problem setup and model introductions are given in Appendix B.1. 

Models/policies merging can improve the performance as shown by the MSE metrics in Table 1 and the scalable training in Fig. 3. Note that our scalable training for ABNets has safety guarantees. The DFB tends to be very conservative as the CBFs within which are not differentiable, which presents a high conservative value shown in Table 1. Our proposed ABNets can significantly reduce the uncertainty of the outputs (controls) under noisy input while guaranteeing safety, and this uncertainty decreases as the increases of the BarrierNet heads in the ABNets, as shown by the last two and three columns in Table 1, as well as shown in Fig. 3 and 6 of Appendix B.1 where the control uncertainty of ABNet-100 is lower than the one of BNet. The smoothness of the controls also increases with the increase of BarrierNet heads (e.g., blue from ABNet v.s. red from BNet in Fig. 6). In terms of performance, our proposed ABNets can also improve the testing errors compared to BNet and DFB, as shown by the MSE in Table 1. The E2Es-MCD model can achieve the best performance, but this is at the cost of safety (the SAFETY metric in Table 1 is negative, which implies violated safety). 


Table 2: Robot manipulation closed-loop testing under noisy input and comparisons with benchmarks.


<table><tr><td>MODEL</td><td>MSE(↓)</td><td>SAFETY (≥0)</td><td>CONSER. (≥0&amp;↓)</td><td><eq>u_1</eq> UNCER-TAINTY (↓)</td><td><eq>u_2</eq> UNCER-TAINTY (↓)</td><td>THEORET. GUAR.</td></tr><tr><td>E2E [18]</td><td>3.6e-4±1.7e-4</td><td>-11.027</td><td>-1.082±2.992</td><td>0.013</td><td>0.009</td><td>×</td></tr><tr><td>E2Es-MCD [12]</td><td>1.1e-4±7.3e-5</td><td>-11.827</td><td>0.162±2.085</td><td>0.008</td><td>0.005</td><td>×</td></tr><tr><td>E2Es-DR [17]</td><td>1.3e-4±8.5e-5</td><td>-11.381</td><td>-0.958±1.875</td><td>0.007</td><td>0.005</td><td>×</td></tr><tr><td>DFB [23]</td><td>8.7e-4±1.9e-4</td><td>2.905</td><td>6.023±3.110</td><td>0.019</td><td>0.018</td><td>√</td></tr><tr><td>BNET [39]</td><td>2.3e-4±1.2e-4</td><td>0.147</td><td>0.745±0.505</td><td>0.010</td><td>0.009</td><td>√</td></tr><tr><td>BNET-UP [36]</td><td>5.2e-5±3.2e-5</td><td>0.206</td><td>0.346±0.098</td><td>0.005</td><td>0.005</td><td>×</td></tr><tr><td>ABNET-10-SC (OURS)</td><td>5.9e-5±5.5e-5</td><td>0.233</td><td>0.570±0.360</td><td>0.006</td><td>0.005</td><td>√</td></tr><tr><td>ABNET-10 (OURS)</td><td>1.2e-4±9.6e-5</td><td>0.039</td><td>0.272±0.443</td><td>0.008</td><td>0.007</td><td>√</td></tr><tr><td>ABNET-100 (OURS)</td><td>1.1e-4±4.4e-5</td><td>0.053</td><td>0.123±0.177</td><td>0.005</td><td>0.004</td><td>√</td></tr></table>

## 4.2 Safe Robot Manipulation

In robot manipulation, we employ a two-link planar robot manipulator to grasp an object from an arbitrary point to an arbitrary destination while avoiding crashing onto obstacles. All the models (h copies/heads) have the same input (with uniformly distributed noise, 10% of the input magnitude in testing). We compare our proposed ABNets with the same benchmark models as in the last subsection. More detailed problem setup and model introductions are given in Appendix B.2. 

Again, models/policies merging can improve the performance as shown by the MSE metrics in Table 2 and the sclable training in Fig. 4. All the E2E-related models are not robust to noise and violate safety constraints (i.e., crash onto obstacles) under noisy input since there are no formal guarantees, and such an example is shown by the magenta trajectory curve of the end-effector in Fig. 4. As shown in Table 2, the proposed ABNet-100 model is the least conservative one with the lowest control uncertainties as well under noisy inputs (significantly improved compared with BNet and DFB), which demonstrates its advantage over other models. This uncertainty improvement is also shown by the control distributions in Fig. 7 in Appendix B.2 (BNet: red area v.s. ABNet-100: blue area). The BNet-UP achieves the best performance without safety guarantees. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-01/69559fc6-5d48-441b-9cdf-fe903d11b9dc/b04dda8036632ac7016b3fdbb3315d951f49665a63cc8f7d9b25d06cddddc0eb.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-01/69559fc6-5d48-441b-9cdf-fe903d11b9dc/cb73424a7a00f610355b47cf81b81cfd6555b9ab2041e5e5a853e794ff8d7fcd.jpg)



log(h): log number of BarrierNet heads with scalable training



Figure 4: Robot manipulation closed-loop end-effector trajectories (left) and ABNet performance with the increasing of BarrierNet heads using scalable training (right). This scalable training for ABNet is with safety guarantees. The transparent red and blue trajectories in the left figure are corresponding to BNet and ABNet-100 models in all runs, respectively.


## 4.3 Vision-based End-to-End Autonomous Driving

We finally test our models in a more complicated and realistic task: vision-based driving, using an open dataset and benchmark from the VISTA [2]. One of ABNets, named ABNet-att, is constructed such that different heads of BarrierNets focus on different parts of the image (left lane boundary, right lane boundary, etc., the corresponding images are shown in Fig 8 of Appendix B.3). For more experiment and model details, please refer to Appendix B.3. 


Table 3: Vision-based end-to-end autonomous driving closed-loop testing and comparisons with benchmarks. New items are short for obstacle crash rate (CRASH), obstacle passing rate (PASS).


<table><tr><td>MODEL</td><td>CRASH (↓)</td><td>PASS (↑)</td><td>SAFETY (≥0)</td><td>CONSER. (≥0&amp;↓)</td><td><eq>u_1</eq> UNCER-TAINTY (↓)</td><td><eq>u_2</eq> UNCER-TAINTY (↓)</td><td>THEORET. GUAR.</td></tr><tr><td>V-E2E [2]</td><td>6%</td><td>94%</td><td>-60.297</td><td>-0.610±21.165</td><td>0.443</td><td>0.222</td><td>×</td></tr><tr><td>E2Es-MCD [12]</td><td>8%</td><td>92%</td><td>-60.566</td><td>-2.211±22.343</td><td>0.429</td><td>0.227</td><td>×</td></tr><tr><td>E2Es-DR [17]</td><td>9%</td><td>91%</td><td>-60.572</td><td>-1.499±21.500</td><td>0.431</td><td>0.224</td><td>×</td></tr><tr><td>DFB [23]</td><td>4%</td><td>39%</td><td>-18.114</td><td>-0.828±5.444</td><td>0.513</td><td>0.125</td><td>√</td></tr><tr><td>BNET [39]</td><td>3%</td><td>33%</td><td>-16.694</td><td>-4.882±4.817</td><td>0.724</td><td>0.385</td><td>√</td></tr><tr><td>BNET-UP [36]</td><td>2%</td><td>35%</td><td>-23.252</td><td>-5.190±4.920</td><td>0.726</td><td>0.532</td><td>×</td></tr><tr><td>ABNET (OURS)</td><td>0%</td><td>100%</td><td>1.455</td><td>6.132±2.181</td><td>0.168</td><td>0.316</td><td>√</td></tr><tr><td>ABNET-ATT (OURS)</td><td>0%</td><td>100%</td><td>4.198</td><td>8.053±1.449</td><td>0.172</td><td>0.269</td><td>√</td></tr><tr><td>ABNET-SC (OURS)</td><td>0%</td><td>100%</td><td>2.221</td><td>7.224±1.667</td><td>0.130</td><td>0.256</td><td>√</td></tr></table>

As shown in Table 3, the proposed ABNets can avoid crash onto obstacles with 100% obstacle passing rate, including the ABNet-sc that is trained in a scalable way with two ABNets (also shown by the scalable training in Fig. 5). This is because the ABNets can learn the correct steering control (the blue and green sine waves shown in Fig. 9 (right) in Appendix B.3) to avoid the obstacle without stopping in front of it. The DFB and BNet-related models learn a significant deceleration control (shown in Fig. 9) to avoid crashing onto obstacles, which explains why the corresponding obstacle passing rates are low compared to other models in Table 3 and why the blue trajectories (BNet) terminate near the obstacle in Fig. 5 (left). Nonetheless, there are still some crash cases in DFB and BNet models due to badly learned CBF parameters that make the inter-sampling effect (i.e., safety violation between discretized times) serious. Most importantly, our proposed ABNet can learn less uncertain controls for this complicated task, as shown in Table 3, the scalable training in Fig. 5, and Fig. 9 (e.g., ABNet:blue or ABNet-att:green area v.s. BNet: red area). The ABNet-att can learn more consistent autonomous driving behavior than the ABNet due to the image attention setting, as shown by the magenta (ABNet-att) and cyan (ABNet) trajectories in Fig. 5 (left) and the green (ABNet-att) and blue (ABNet) areas in Fig. 9. Ablation studies on the robustness of our ABNets in terms of safety under high-noisy inputs (50% noise level) are given in Table 4 of Appendix B.3. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-01/69559fc6-5d48-441b-9cdf-fe903d11b9dc/af8e3fa7a5a77b14ab3f920ded57e8162e59d1e5582e70ac47daf2bc019693ad.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-01/69559fc6-5d48-441b-9cdf-fe903d11b9dc/a82c1ee9bd53851103d1f953a17f501b6b49efa9de98af166a94f688d08e3086.jpg)



h: number of BarrierNet heads with scalable training



Figure 5: Vision-based end-to-end autonomous driving closed-loop testing trajectories in VISTA (left) and ABNet performance with the increasing of BarrierNet heads using scalable training (right). This scalable training is done by both the ABNet and ABNet-att in Table 3 with safety guarantees.


## 5 Related Works

Scalability, merging and uncertainty in learning for robot control. Machine learning techniques have been widely used in robot control [7] [32] [35]. Mixture of expert methods [31] [27] [42] are scalable but hard to retain the property (such as safety) of the models. The uncertainty resulting from noisy model input or dataset is preventing the deployment to real robots [21] [15]. To address this, predictive uncertainty quantification [12] [17], also a model merging approach, has been widely adopted. It has been shown to work well in vision-based autonomous driving under noisy input [36] using the Gaussian kernel with Scott’s rule [30] to select bandwidth. The main challenge of this technique is that it may make the system lose performance guarantees, such as safety. Other model merging approaches [14] [26] [34] do not preserve safety either. We address the uncertainty and scalablibity problem for robot control using the proposed ABNets with provable safety guarantees. 

CBFs and set invariance. In control theory, the set invariance has been widely adopted to prove and enforce the safety of dynamical systems [6] [25] [1] [38] [39]. The Control Barrier Function (CBF) [1] [38] is such a state of the art technique that can enforce set invariance [5], [24], [37], and transforms a nonlinear optimization problem to a quadratic problem that is very efficient to solve. CBFs originates from barrier functions that are originally used in optimization problems [8]. However, the CBF method tends to make the system conservative (i.e., at the cost of performance) in order to enforce safety, and it is not scalable to build large safe filters in neural networks. Our proposed ABNet can address all these limitations. 

Safety in neural networks. Safety is usually enforced using optimizations. Recently, differentiable optimizations show great potential for learning-based control with safety guarantees [23, 4, 39, 20]. The quadratic program (QP) can be employed as a layer in the neural network, i.e., the OptNet [3]. The OptNet has been used with CBFs in neural networks as a safe filter controls [23], in which CBFs themselves are not trainable, which can significantly limiting the learning capability. Neural network controllers with safety certificate have been learned through verification-in-the-loop training [10, 41, 11]. However, this verification method cannot ensure to cover the whole state space. CBFs are also used in neural ODEs to equip them with specification guarantees [40]. None of these methods are scalable to larger models, and are subject to uncertainty, which the proposed ABNet can address. 

## 6 Conclusions, Limitations and Future Work

We propose a novel Attention BarrierNet (ABNet) that merge many safety-critical learning models while preserving the safety in this paper. The proposed ABNet is scalable to larger safe learning models, can achieve better performance, and is robust to input noise. We have demonstrated the effectiveness of the model on a series of robot control tasks. Nonetheless, our model (and all the other barrier-based learning models [11] [39]) still have a few limitations motivating for further research. 

Limitations. First, the ABNet depends on the system/robot dynamics to strictly enforce safety guarantees. We may use neural ODEs [9] to simultaneously learn the dynamics in the ABNet if they are unknown. Second, the ABNet also depends on accurate system/robot state that is hard to estimate from high-dimensional observations. We will explore to use foundation models [19] in conjunction with ABNet to address such a challenge in the future. Finally, the ABNet also requires safety specifications that may be unknown in some robot control tasks, we may learn the safety specifications from data [28], [33], and this can also be done in conjunction with ABNet. 

## 7 Acknowledgement

The research was supported in part by Capgemini Engineering. It was also partially sponsored by the United States Air Force Research Laboratory and the United States Air Force Artificial Intelligence Accelerator and was accomplished under Cooperative Agreement Number FA8750-19-2- 1000. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the United States Air Force or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein. This research was also supported in part by the AI2050 program at Schmidt Futures (Grant G- 965 22-63172). 

## References



[1] A. D. Ames, X. Xu, J. W. Grizzle, and P. Tabuada. Control barrier function based quadratic programs for safety critical systems. IEEE Transactions on Automatic Control, 62(8):3861–3876, 2017. 





[2] Alexander Amini, Tsun-Hsuan Wang, Igor Gilitschenski, Wilko Schwarting, Zhijian Liu, Song Han, Sertac Karaman, and Daniela Rus. Vista 2.0: An open, data-driven simulator for multimodal sensing and policy learning for autonomous vehicles. In 2022 International Conference on Robotics and Automation (ICRA), pages 2419–2426. IEEE, 2022. 





[3] Brandon Amos and J. Zico Kolter. Optnet: Differentiable optimization as a layer in neural networks. In Proceedings of the 34th International Conference on Machine Learning - Volume 70, pages 136–145, 2017. 





[4] Brandon Amos, Ivan Dario Jimenez Rodriguez, Jacob Sacks, Byron Boots, and J. Zico Kolter. Differentiable mpc for end-to-end planning and control. In Proceedings of the 32nd International Conference on Neural Information Processing Systems, page 8299–8310. Curran Associates Inc., 2018. 





[5] Jean-Pierre Aubin. Viability theory. Springer, 2009. 





[6] Franco Blanchini. Set invariance in control. Automatica, 35(11):1747–1767, 1999. 





[7] Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al. On the opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258, 2021. 





[8] S. P. Boyd and L. Vandenberghe. Convex optimization. Cambridge university press, New York, 2004. 





[9] Ricky TQ Chen, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud. Neural ordinary differential equations. In Proceedings of the 32nd International Conference on Neural Information Processing Systems, pages 6572–6583, 2018. 





[10] Jyotirmoy V. Deshmukh, James P. Kapinski, Tomoya Yamaguchi, and Danil Prokhorov. Learning deep neural network controllers for dynamical systems with safety guarantees: Invited paper. In 2019 IEEE/ACM International Conference on Computer-Aided Design (ICCAD), pages 1–7, 2019. 





[11] James Ferlez, Mahmoud Elnaggar, Yasser Shoukry, and Cody Fleming. Shieldnn: A provably safe nn filter for unsafe nn controllers. preprint arXiv:2006.09564, 2020. 





[12] Yarin Gal and Zoubin Ghahramani. Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In international conference on machine learning, pages 1050–1059. PMLR, 2016. 





[13] P. Glotfelter, J. Cortes, and M. Egerstedt. Nonsmooth barrier functions with applications to multi-robot systems. IEEE control systems letters, 1(2):310–315, 2017. 





[14] Chengsong Huang, Qian Liu, Bill Yuchen Lin, Tianyu Pang, Chao Du, and Min Lin. Lorahub: Efficient cross-task generalization via dynamic lora composition. arXiv preprint arXiv:2307.13269, 2023. 





[15] Gregory Kahn, Adam Villaflor, Vitchyr Pong, Pieter Abbeel, and Sergey Levine. Uncertainty-aware reinforcement learning for collision avoidance. arXiv preprint arXiv:1702.01182, 2017. 





[16] Hassan K. Khalil. Nonlinear Systems. Prentice Hall, third edition, 2002. 





[17] Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. Advances in neural information processing systems, 30, 2017. 





[18] Sergey Levine, Chelsea Finn, Trevor Darrell, and Pieter Abbeel. End-to-end training of deep visuomotor policies. Journal of Machine Learning Research, 17(39):1–40, 2016. 





[19] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International conference on machine learning, pages 12888–12900. PMLR, 2022. 





[20] Wenliang Liu, Wei Xiao, and Calin Belta. Learning robust and correct controllers from signal temporal logic specifications using barriernet. In 2023 62nd IEEE Conference on Decision and Control (CDC), pages 7049–7054. IEEE, 2023. 





[21] Antonio Loquercio, Mattia Segu, and Davide Scaramuzza. A general framework for uncertainty estimation in deep learning. IEEE Robotics and Automation Letters, 5(2):3153–3160, 2020. 





[22] Mitio Nagumo. Über die lage der integralkurven gewöhnlicher differentialgleichungen. In Proceedings of the Physico-Mathematical Society of Japan. 3rd Series. 24:551-559, 1942. 





[23] Marcus Aloysius Pereira, Ziyi Wang, Ioannis Exarchos, and Evangelos A. Theodorou. Safe optimal control using stochastic barrier functions and deep forward-backward sdes. In Conference on Robot Learning, 2020. 





[24] Stephen Prajna, Ali Jadbabaie, and George J. Pappas. A framework for worst-case and stochastic safety verification using barrier certificates. IEEE Transactions on Automatic Control, 52(8):1415–1428, 2007. 





[25] Sasa V Rakovic, Eric C Kerrigan, Konstantinos I Kouramas, and David Q Mayne. Invariant approximations of the minimal robust positively invariant set. IEEE Transactions on automatic control, 50(3):406–410, 2005. 





[26] Alexandre Ramé, Kartik Ahuja, Jianyu Zhang, Matthieu Cord, Léon Bottou, and David Lopez-Paz. Model ratatouille: Recycling diverse models for out-of-distribution generalization. In International Conference on Machine Learning, pages 28656–28679. PMLR, 2023. 





[27] Carlos Riquelme, Joan Puigcerver, Basil Mustafa, Maxim Neumann, Rodolphe Jenatton, André Susano Pinto, Daniel Keysers, and Neil Houlsby. Scaling vision with sparse mixture of experts. Advances in Neural Information Processing Systems, 34:8583–8595, 2021. 





[28] Alexander Robey, Haimin Hu, Lars Lindemann, Hanwen Zhang, Dimos V. Dimarogonas, Stephen Tu, and Nikolai Matni. Learning control barrier functions from expert demonstrations. In 2020 59th IEEE Conference on Decision and Control (CDC), pages 3717–3724, 2020. 





[29] Alessandro Rucco, Giuseppe Notarstefano, and John Hauser. An efficient minimum-time trajectory generation strategy for two-track car vehicles. IEEE Transactions on Control Systems Technology, 23(4):1505– 1519, 2015. 





[30] David W Scott. Multivariate density estimation: theory, practice, and visualization. John Wiley & Sons, 2015. 





[31] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538, 2017. 





[32] Ishika Singh, Valts Blukis, Arsalan Mousavian, Ankit Goyal, Danfei Xu, Jonathan Tremblay, Dieter Fox, Jesse Thomason, and Animesh Garg. Progprompt: Generating situated robot task plans using large language models. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 11523–11530. IEEE, 2023. 





[33] M. Srinivasan, A. Dabholkar, S. Coogan, and P. A. Vela. Synthesis of control barrier functions using a supervised machine learning approach. In 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 7139–7145, 2020. 





[34] Lirui Wang, Jialiang Zhao, Yilun Du, Edward H Adelson, and Russ Tedrake. Poco: Policy composition from and for heterogeneous robot learning. arXiv preprint arXiv:2402.02511, 2024. 





[35] Tsun-Hsuan Wang, Alaa Maalouf, Wei Xiao, Yutong Ban, Alexander Amini, Guy Rosman, Sertac Karaman, and Daniela Rus. Drive anywhere: Generalizable end-to-end autonomous driving with multi-modal foundation models. arXiv preprint arXiv:2310.17642, 2023. 





[36] Tsun-Hsuan Wang, Wei Xiao, Makram Chahine, Alexander Amini, Ramin Hasani, and Daniela Rus. Learning stability attention in vision-based end-to-end driving policies. In Proceedings of The 5th Annual Learning for Dynamics and Control Conference, volume 211 of Proceedings of Machine Learning Research, pages 1099–1111. PMLR, 15–16 Jun 2023. 





[37] Rafael Wisniewski and Christoffer Sloth. Converse barrier certificate theorem. In Proc. of 52nd IEEE Conference on Decision and Control, pages 4713–4718, Florence, Italy, 2013. 





[38] Wei Xiao and Calin Belta. High-order control barrier functions. IEEE Transactions on Automatic Control, 67(7):3655–3662, 2022. 





[39] Wei Xiao, Tsun-Hsuan Wang, Ramin Hasani, Makram Chahine, Alexander Amini, Xiao Li, and Daniela Rus. Barriernet: Differentiable control barrier functions for learning of safe robot control. IEEE Transac tions on Robotics, 2023. 





[40] Wei Xiao, Tsun-Hsuan Wang, Ramin Hasani, Mathias Lechner, Yutong Ban, Chuang Gan, and Daniela Rus. On the forward invariance of neural odes. In International conference on machine learning, pages 38100–38124. PMLR, 2023. 





[41] Hengjun Zhao, Xia Zeng, Taolue Chen, Zhiming Liu, and Jim Woodcock. Learning safe neural network controllers with barrier certificates. Form Asp Comp, 33:437–455, 2021. 





[42] Yanqi Zhou, Tao Lei, Hanxiao Liu, Nan Du, Yanping Huang, Vincent Zhao, Andrew M Dai, Quoc V Le, James Laudon, et al. Mixture-of-experts with expert choice routing. Advances in Neural Information Processing Systems, 35:7103–7114, 2022. 



## A Proof of Theorems

Theorem 3.1. (Safety of ABNets) Given the multi-head BarrierNets formulated as in (6) s.t. (7). If the system (1) is initially safe $( \mathrm { i . e . , } b _ { j } ( \pmb { x } ( t _ { 0 } ) ) \geq 0 , \forall j \in S )$ , then a control policy u from the ABNet output (8) guarantees the safety of system $( 1 ) , \mathrm { i . e . , } \bar { b _ { j } } ( { \pmb x } ( t ) ) \geq 0 , \forall j \in S , \forall \bar { t } \geq t _ { 0 }$ 

Proof: The proof outline is to first show the existence of new HOCBF constraints (corresponding to all the safety specifications) that are defined over the output of the ABNet. Then, we can use Nagumo’s theorem [22] to recursively show the forward invariance of each safety set in the HOCBFs, and this can eventually imply the satisfaction of the safety specifications $b _ { j } ( { \pmb x } ) \overset { \cdot } { \geq } 0 , \forall j \in S$ 

Since each control ${ \pmb u } _ { k } , k \in \{ 1 , \ldots , h \}$ in the ABNet is obtained from solving the $\mathrm { Q P } \left( 6 \right) \ \mathrm { s . t . } \ ( 7 )$ , we have that the following constraint is satisfied: 

$$
L _ {f} \psi_ {j, m - 1} (\boldsymbol {x}, \boldsymbol {z} | \theta_ {p}) + [ L _ {g} \psi_ {j, m - 1} (\boldsymbol {x}, \boldsymbol {z} | \theta_ {p}) ] \boldsymbol {u} _ {k} + p _ {m, k} (\boldsymbol {z} _ {k} | \theta_ {p, k} ^ {m}) \alpha_ {j, m} (\psi_ {j, m - 1} (\boldsymbol {x}, \boldsymbol {z} | \theta_ {p})) \geq 0, j \in S,\tag{10}
$$

Multiplying the weight $w _ { k } \ge 0$ to the last equation, we have 

$$
w _ {k} L _ {f} \psi_ {j, m - 1} (\boldsymbol {x}, \boldsymbol {z} | \theta_ {p}) + w _ {k} \left[ L _ {g} \psi_ {j, m - 1} (\boldsymbol {x}, \boldsymbol {z} | \theta_ {p}) \right] \boldsymbol {u} _ {k} + w _ {k} p _ {m, k} \left(\boldsymbol {z} _ {k} \mid \theta_ {p, k} ^ {m}\right) \alpha_ {j, m} \left(\psi_ {j, m - 1} (\boldsymbol {x}, \boldsymbol {z} | \theta_ {p})\right) \geq 0, j \in S,\tag{11}
$$

Taking a summation of the last equation over all $k \in \{ 1 , \ldots , h \}$ , the following equation establishes: 

$$
\begin{array}{l} \sum_ {k = 1} ^ {h} w _ {k} L _ {f} \psi_ {j, m - 1} (\boldsymbol {x}, \boldsymbol {z} | \boldsymbol {\theta} _ {p}) + \sum_ {k = 1} ^ {h} w _ {k} [ L _ {g} \psi_ {j, m - 1} (\boldsymbol {x}, \boldsymbol {z} | \boldsymbol {\theta} _ {p}) ] \boldsymbol {u} _ {k} \\ + \sum_ {k = 1} ^ {h} w _ {k} p _ {m, k} (\boldsymbol {z} _ {k} | \boldsymbol {\theta} _ {p, k} ^ {m}) \alpha_ {j, m} (\psi_ {j, m - 1} (\boldsymbol {x}, \boldsymbol {z} | \boldsymbol {\theta} _ {p})) \geq 0, j \in S, \end{array}\tag{12}
$$

Since $L _ { g } \psi _ { j , m - 1 } ( { \pmb x } , z | \theta _ { p }$ is a vector that is independent of k and $\begin{array} { r } { \sum _ { k = 1 } ^ { h } w _ { k } = 1 } \end{array}$ , the last equation can be rewritten as: 

$$
L _ {f} \psi_ {j, m - 1} (\boldsymbol {x}, \boldsymbol {z} | \theta_ {p}) + L _ {g} \psi_ {j, m - 1} (\boldsymbol {x}, \boldsymbol {z} | \theta_ {p}) \left(\sum_ {k = 1} ^ {h} w _ {k} \boldsymbol {u} _ {k}\right)\tag{13}
$$

$$
+ \sum_ {k = 1} ^ {h} w _ {k} p _ {m, k} \left(\boldsymbol {z} _ {k} \mid \theta_ {p, k} ^ {m}\right) \alpha_ {j, m} \left(\psi_ {j, m - 1} (\boldsymbol {x}, \boldsymbol {z} \mid \theta_ {p})\right) \geq 0, j \in S,
$$

The summation of class $\mathcal { K }$ functions is also a class $\mathcal { K }$ function. Since $\alpha _ { j , m }$ are class $\mathcal { H }$ functions, the $\begin{array} { r } { \sum _ { k = 1 } ^ { h } w _ { k } p _ { m , k } ( z _ { k } | \theta _ { p , k } ^ { m } ) \alpha _ { j , m } ( \psi _ { j , m - 1 } ( { \pmb x } , z | \theta _ { p } ) ) } \end{array}$ ) is also a class $\mathcal { K }$ function over $\psi _ { j , m - 1 } \mathopen { } \mathclose \bgroup \left( \pmb { x } , z \aftergroup \egroup | \theta _ { p } \aftergroup \egroup \right)$ Therefore, equations (13) are the new HOCBF constraints defined over the output of the ABNet, $\begin{array} { r } { { \mathrm { i } } . { \mathrm { e } } . , \sum _ { k = 1 } ^ { h } w _ { k } { \mathbf { } } { \mathbf { } } u _ { k } } \end{array}$ . In other words, whenever $\psi _ { j , m - 1 } ( { \pmb x } , z | \theta _ { p } ) = 0$ , we have 

$$
L _ {f} \psi_ {j, m - 1} (\boldsymbol {x}, \boldsymbol {z} | \theta_ {p}) + L _ {g} \psi_ {j, m - 1} (\boldsymbol {x}, \boldsymbol {z} | \theta_ {p}) \left(\sum_ {k = 1} ^ {h} w _ {k} \boldsymbol {u} _ {k}\right) \geq 0, j \in S,\tag{14}
$$

The controls (outputs of the ABNet) $\begin{array} { r } { \sum _ { k = 1 } ^ { h } w _ { k } { \pmb u } _ { k } \equiv { \pmb u } } \end{array}$ are directly used to drive the system (1), and z is taken as a piece-wise constant within discretized time intervals [39]. Therefore, the last equation can be rewritten as 

$$
\frac {\partial \psi_ {j , m - 1} (\boldsymbol {x} , \boldsymbol {z} | \theta_ {p})}{\partial \boldsymbol {x}} (f (\boldsymbol {x}) + g (\boldsymbol {x}) \boldsymbol {u}) = \frac {\partial \psi_ {j , m - 1} (\boldsymbol {x} , \boldsymbol {z} | \theta_ {p})}{\partial \boldsymbol {x}} \dot {\boldsymbol {x}} = \dot {\psi} _ {j, m - 1} (\boldsymbol {x}, \boldsymbol {z} | \theta_ {p}) \geq 0, j \in S,\tag{15}
$$

Since $b _ { j } ( { \pmb x } ( t _ { 0 } ) ) \geq 0$ , we can always initialize the HOCBF definition such that $\dot { \psi } _ { j , m - 1 } ( { \pmb x } , z | \theta _ { p } ) \geq 0$ is satisfied at t<sub>0</sub> [38]. By Nagumo’s theorem [22] and (13)-(15), we have that $\psi _ { j , m - 1 } ( \pmb { x } , z | \pmb { \theta } _ { p } ) \geq 0 , \forall t \geq$ $t _ { 0 } .$ 

Recursively, we can show that $\psi _ { j , i } ( \pmb { x } , z | \theta _ { p } ) \geq 0 , \forall t \geq t _ { 0 } , \forall i \in \{ 0 , . . . , m - 1 \}$ from $i = m - 1 { \mathrm { ~ t o ~ } } i = 0$ Since $b _ { j } ( { \pmb x } ) = \Psi _ { j , 0 } ( { \pmb x } , z | \theta _ { p } )$ by (2), we have that $b _ { j } ( { \pmb x } ( t ) ) \geq 0 , \forall t \geq t _ { 0 } , \forall j \in S$ , which the safety guarantees of the ABNet for system (1). ■ 

Theorem 3.2. (Safety of merging of ABNets) Given two ABNets with each formulated as in (8) and (6) s.t. (7), the merged model using the form as in (8) again guarantees the safety of system (1). 

Proof: The proof outline is similar to that of Theorem 3.1. From each ABNet, we can show the existence of new HOCBF constraints (corresponding to all the safety specifications) that are defined over the output of each ABNet. Then we can again show the existence of another set of new HOCBF constraints (corresponding to all the safety specifications) that are defined over the output of the merged ABNet. Finally, we can also use Nagumo’s theorem [22] to recursively show the forward invariance of each safety set in the HOCBFs, and this can eventually imply the satisfaction of the safety specifications $b _ { j } ( \mathbf { \bar { x } } ) \geq 0 , \forall j \in S .$ 

The mathematical proof is similar to that of Theorem 3.1, and thus is omitted. 

## B Experiment Details

Metrics used in all the tables. The SAFETY metric is defined as: 

$$
\text { SAFETY } = \min _ {k} \bigl \{\min _ {t \in [ t _ {0}, T ]} b (\boldsymbol {x} (t) \bigr \} _ {k}, k \in \{1, \dots , N \},\tag{16}
$$

where N is the number of testing runs $( N = 1 0 0$ in this case). T is the final time of each run. $b ( { \pmb x } ) \geq 0$ is the safety constraint that is given explicitly in each experiment below. 

The CONSER. metric is defined as 

$$
\begin{array}{c} \text { CONSER.   mean } = \underset {k} {\text { mean }} \bigl \{\underset {t \in [ t _ {0}, T ]} {\min} b (\boldsymbol {x} (t) \bigr \} _ {k}, k \in \{1, \ldots , N \}, \\ \text { CONSER.   std } = \underset {k} {\text { std }} \bigl \{\underset {t \in [ t _ {0}, T ]} {\min} b (\boldsymbol {x} (t) \bigr \} _ {k}, k \in \{1, \ldots , N \}. \end{array}\tag{17}
$$

The UNCERTAINTY metric for both controls are calculated by: 

$$
u _ {i} \text {   UNCERTAINTY } = \underset {t \in [ t _ {0}, T ]} {\text { mean }} \left\{\underset {k} {\text { std }} \{u _ {i} (t) \} _ {k}, k \in \{1, \dots , N \} \right\}, i \in \{1, 2 \}.\tag{18}
$$

## B.1 2D Robot Obstacle Avoidance

Models. All the models include fully connected layers of shape [5, 128, 32, 32, 2] with RELU as activation functions. There are some additional layers of differentiable QPs in other models (other than E2E-related models). The model input is the system state and the goal. 

Training and Dataset. The dataset includes 100 trajectories, and each trajectory has 137 trajectory points. The ground truth controls (i.e., training labels) are obtained via solving HOCBF-based QPs [38]. We use Adam as the optimizer to train the model with a MSE loss function and a learning rate 0.001. We use the QPFunction from the OptNet [3] to solve the dQPs. The training time of the ABNet is about 1 hour for 20 epochs on a RTX-3090 computer. 

Robot dynamics and safety constraints. We employ the bicycle model as the robot dynamics: 

$$
\underbrace {\left[ \begin{array}{c} \dot {x} (t) \\ \dot {y} (t) \\ \dot {\theta} (t) \\ \dot {v} (t) \end{array} \right]} _ {\dot {\boldsymbol {x}} (t)} = \underbrace {\left[ \begin{array}{c} v (t) \cos \theta (t) \\ v (t) \sin \theta (t) \\ 0 \\ 0 \end{array} \right]} _ {f (\boldsymbol {x})} + \underbrace {\left[ \begin{array}{c c} 0 & 0 \\ 0 & 0 \\ 1 & 0 \\ 0 & 1 \end{array} \right]} _ {g (\boldsymbol {x})} \underbrace {\left[ \begin{array}{c} u _ {1} (t) \\ u _ {2} (t) \end{array} \right]} _ {\boldsymbol {u}}\tag{19}
$$

where $( x , y ) \in \mathbb { R } ^ { 2 }$ denotes the 2D location of the robot, $\theta \in \mathbb { R }$ is the heading angle of the robot, $\nu \in \mathbb { R }$ is the linear speed of the robot. $u _ { 1 } , u _ { 2 }$ are the angular speed and acceleration controls, respectively. The safety constraint of the robot is defined as: 

$$
b (\boldsymbol {x}) = (x - x _ {0}) ^ {2} + (y - y _ {0}) ^ {2} - R ^ {2} \geq 0,\tag{20}
$$

where $( x _ { 0 } , y _ { 0 } ) \in \mathbb { R } ^ { 2 }$ is the 2D location of the obstacle, and $R > 0$ is its size. 

Acceleration control profiles. We show the acceleration control profiles in Fig. 6. The corresponding uncertainty is also significantly decreased with the proposed ABNet. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-01/69559fc6-5d48-441b-9cdf-fe903d11b9dc/56ceb2cadad7798d5198a5c668dfef91df8cf38c0fd6786c36ba92cca01c7d6b.jpg)



Figure 6: 2D robot obstacle avoidance acceleration control profiles and their distributions. The controls are subject to input noise, and thus are non-smooth. All the testings are done in a closed-loop fashion, i.e., the model outputs are directly used to control the robot.


## B.2 Safe Robot Manipulation

Models. All the models include fully connected layers of shape [6, 128, 256, 128, 128, 32, 32, 2] with RELU as activation functions. There are some additional layers of differentiable QPs in other models (other than E2E-related models). The model input is the system state and the goal. 

Training and Dataset. The dataset includes 1000 trajectories, and each trajectory has about 350 trajectory points. The ground truth controls (i.e., training labels) are obtained via solving HOCBFbased QPs [38]. We use Adam as the optimizer to train the model with a MSE loss function and a learning rate 0.001. We use the QPFunction from the OptNet [3] to solve the dQPs. The training time of the ABNet is about 2 hours for 10 epochs on a RTX-3090 computer. 

Robot dynamics and safety constraints. We employ the following model as the manipulator dynamics: 

$$
\underbrace {\left[ \begin{array}{c} \dot {\theta} _ {1} \\ \dot {\omega} _ {1} \\ \dot {\theta} _ {2} \\ \dot {\omega} _ {2} \end{array} \right]} _ {\boldsymbol {x}} = \underbrace {\left[ \begin{array}{c} \omega_ {1} \\ 0 \\ \omega_ {2} \\ 0 \end{array} \right]} _ {f (\boldsymbol {x})} + \underbrace {\left[ \begin{array}{c c} 0 & 0 \\ 1 & 0 \\ 0 & 0 \\ 0 & 1 \end{array} \right]} _ {g (\boldsymbol {x})} \underbrace {\left[ \begin{array}{c} u _ {1} \\ u _ {2} \end{array} \right]} _ {\boldsymbol {u}}\tag{21}
$$

where $( \theta _ { 1 } , \theta _ { 2 } ) \in \mathbb { R } ^ { 2 }$ denotes the angles of the two-link manipulator joints, $( \omega _ { 1 } , \omega _ { 2 } ) \in \mathbb { R } ^ { 2 }$ is the angular speed of the two-link manipulator joints, u , u are the angular acceleration controls corresponding to the two joints, respectively. 

The safety constraint of the robot is defined as: 

$$
b (\boldsymbol {x}) = \left(l _ {1} \cos \theta_ {1} + l _ {2} \cos \theta_ {2} - x _ {0}\right) ^ {2} + \left(l _ {1} \sin \theta_ {1} + l _ {2} \sin \theta_ {2} - y _ {0}\right) ^ {2} - R ^ {2} \geq 0,\tag{22}
$$

where $( x _ { 0 } , y _ { 0 } ) \in \mathbb { R } ^ { 2 }$ is the location of the obstacle, and $R > 0$ is its size. $l _ { 1 } > 0 , l _ { 2 } > 0$ are the length of the two links of the manipulator, respectively. In the current setting, the non-collision of the end-effector implies the non-collision of the link. Therefore, we only need to consider the safety of the end-effector. We show both the $u _ { 1 } , u _ { 2 }$ control profiles in Fig. 7 to demonstrate the advantage of the proposed ABNet. The metric definitions are the same as in the 2D robot obstacle avoidance, and the number of testing runs is N = 100. 

## B.3 Vision-based End-to-End Autonomous Driving

Models. All the models include CNN ([[3, 24, 5, 2, 2], [24, 36, 5, 2, 2], [36, 48, 3, 2, 1], [48, 64, 3, 1, 1], [64, 64, 3, 1, 1]]) and LSTM layers (size: 64) and some fully connected layers of shape [32, 32, 2] ×2 with RELU as activation functions. The dropout rates for both CNN and fully connected layers are 0.3. There are some additional layers of differentiable QPs in other models (other than E2E-related models). The model input is the front-view RGB images (shape: $3 \times 4 5 \times 1 5 5 )$ of the ego vehicle, and the outputs are the steering rate and acceleration controls of the vehicle. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-01/69559fc6-5d48-441b-9cdf-fe903d11b9dc/e3d78fe00dde500da668f81a1ed748b18c501afdba91933e91c01e155bf4aeaf.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-01/69559fc6-5d48-441b-9cdf-fe903d11b9dc/3d4a22273888b66d3aa9127ed0ff0ce2565d9c8a36f37e2afb1b2ae89c69e2e8.jpg)



Figure 7: Robot manipulation joint control profiles and their distributions. The controls are subject to input noise, and thus are non-smooth. All the testings are done in a closed-loop fashion, i.e., the model outputs are directly used to control the manipulator.


Training and Dataset. The dataset is open-sourced including 0.4 million image-control pairs from a closed-road sim-to-real driving field. Static and parked cars of different types and colors are used as obstacles in the dataset. The dataset is collected from the VISTA simulator [2]. The ground truth controls (i.e., training labels) are obtained via solving a nonlinear model predictive control (NMPC). We use Adam as the optimizer to train the model with a MSE loss function and a learning rate 0.001. We use the QPFunction from the OptNet [3] to solve the dQPs. The training time of the ABNet is about 15 hours for 5 epochs on a RTX-3090 computer. 

Brief introduction to VISTA. VISTA is a sim-to-real driving simulator that can generate driving scenarios from real driving data [2]. The VISTA allows us to train our model with guided policy learning. This learning method has been shown to work for model transfer to a full-scale real autonomous vehicle. There three steps to generate the data: (i) In VISTA, we randomly initialize the locations and poses of ego- and ado-cars that are associated with the real driving data; (ii) we use NMPC to collect ground-truth controls (training labels) with corresponding states, and (iii) we collect front-view RGB images along the trajectories generated from NMPC. 

Vehicle dynamics and safety constraints. The vehicle dynamics are specified with respect to a reference trajectory [29], such as the lane center line. The two most important states are the alongtrajectory progress $s \in \mathbb { R }$ and the lateral offset distance $d \in \mathbb { R }$ of the vehicle center with respect to the trajectory. The dynamics are defined as: 

$$
\underbrace {\left[ \begin{array}{c} \dot {s} \\ \dot {d} \\ \dot {\mu} \\ \dot {v} \\ \dot {\delta} \end{array} \right]} _ {\dot {\boldsymbol {x}}} = \underbrace {\left[ \begin{array}{c} \frac {v \cos (\mu + \beta)}{1 - d \kappa} \\ v \sin (\mu + \beta) \\ \frac {v}{l _ {r}} \sin \beta - \kappa \frac {v \cos (\mu + \beta)}{1 - d \kappa} \\ 0 \\ 0 \end{array} \right]} _ {f (\boldsymbol {x})} + \underbrace {\left[ \begin{array}{c c} 0 & 0 \\ 0 & 0 \\ 0 & 0 \\ 1 & 0 \\ 0 & 1 \end{array} \right]} _ {g (\boldsymbol {x})} \underbrace {\left[ \begin{array}{c} u _ {1} \\ u _ {2} \end{array} \right]} _ {\boldsymbol {u}},\tag{23}
$$

where $\mu$ is the local heading error of the vehicle with respect to the reference trajectory, v is the linear speed of the vehicle, κ is the curvature of the trajectory at the progess s. $l _ { r }$ is the length of the vehicle from the tail to the center, β = arctan $\left( \frac { l _ { r } } { l _ { r } + l _ { f } } \tan \delta \right)$ , where $l _ { f }$ is the length of the vehicle from the head to the center. $u _ { 1 } , u _ { 2 }$ are the steering rate and acceleration controls of the vehicle, respectively. The safety constraint of the vehicle is defined as: 

$$
b (\boldsymbol {x}) = (s - s _ {0}) ^ {2} + (d - d _ {0}) ^ {2} - R ^ {2} \geq 0,\tag{24}
$$

where $( s _ { 0 } , d _ { 0 } ) \in \mathbb { R } ^ { 2 }$ is the location of the obstacle in the curvi-linear frame (i.e., defined with respect to the reference trajectory), and $R > 0$ defines its size that is chosen such that the satisfaction of the above constraint can make the ego vehicle avoid crashing onto the obstacle. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-01/69559fc6-5d48-441b-9cdf-fe903d11b9dc/32d753b3f1e9d29e3710cfe295cebdeb9e86aa572bd0db34417a6883b83ae9e4.jpg)



Figure 8: Attention-based image observations for the ABNet-att model. From left to right and top to down: attentions on full image, left-most part, left lane boundary, lane center, right lane boundary, and right-most part.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-01/69559fc6-5d48-441b-9cdf-fe903d11b9dc/bf1858a575c35cd8e4e43664ec37955b40744857385d2cf4319fa4ea0bac615e.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-01/69559fc6-5d48-441b-9cdf-fe903d11b9dc/f0c5289ddd53f06fe20729ae0b67a8500db83f4392ed706a973d1083899f653e.jpg)



Figure 9: Vision-based end-to-end autonomous driving closed-loop testing control profiles. The models directly take images as inputs, and output controls for the vehicle. All the testings are done in closed-loop in VISTA.


Closed-loop testing. We test all of our models in a closed-loop manner in VISTA. In other words, at each time step, we get the front-view RGB image observation from VISTA. Then, the model generates a control based on the image. Finally, the control is used to drive the “virtual” vehicle in VISTA. This process is done recursively until the final time. The total number of testing runs is $N = 1 0 0$ for all the tables. The obstacles are randomly initialized (in uniform probability distribution) with lateral distance $d _ { 0 }$ ranges from ±0.1m to ±1.5m. In Figs. 5 and 9, the ego vehicle is randomly initialized with $d \in [ - 0 . 5 , 0 . 5 ] m$ (in uniform probability distribution). 

Image observations for the ABNet-att model. We generate the attention-based observations as shown in Fig. 8. Each of the attention images may play an important role in a specific driving scenario (e.g., attention on the left-most part may be crucial for sharp-left turn). 

Acceleration control profiles. We present both the acceleration control and steering rate control profiles in Fig. 9. Both the BNet and BNet-UP models have forced the ego vehicle to have a large deceleration instead of making it to pass the obstacle using the steering control when the vehicle approaches the obstacle. This can make the ego vehicle get stuck at the obstacles, and thus, the obstacle passing rate (as shown in Table 3) is low in these two models. 

Ablation studies on the model robustness in terms of safety under noisy input. To further test the model safety robustness, we add random noise (50% magnitude of the image values) to all the image observations. The results are presented in Table 4. Our proposed ABNets can still guarantee the safety of the vehicle under noisy input (0% crash rate), while the crash rates using other models significantly increase except the DFB model. This is because the HOCBFs in the DFB model are not trainable, and the corresponding parameters are fixed. Badly trained HOCBFs could make the method fail to guarantee safety due to the inter-sampling effect. 


Table 4: Ablation study: vision-based end-to-end autonomous driving closed-loop testing under noise and comparisons with benchmarks. Items in the first row are short for obstacle crash rate (CRASH), Obstacle passing rate (PASS), satisfaction of safety constraints where non-negative values mean safety guarantees (SAFETY), system conservativeness (CONSER.), acceleration control $u _ { 1 }$ uncertainty $( u _ { 1 }$ UNCERTAINTY), steering rate control u uncertainty (u UNCERTAINTY), and theoretical safety guarantees (THEORET. GUAR.) respectively. In the model column, items are short for single vanilla end-to-end driving model (V-E2E), E2Es merged with Monte-Carlo Dropout (E2Es-MCD), E2Es merged with deep resembles (E2Es-MERG), deep forward and backward model (DFB), single BarrierNet (BNET), BarrierNet policies with uncertainty propagation (BNET-UP), ABNet with 10 heads (ABNET), ABNet with attention images and 10 heads (ABNET-ATT), ABNET-SC denotes our ABNet first trained with ABNET-ATT scaled by ABNET (20 heads)respectively. The safety metric is defined as the minimum value of the safety specification $b _ { j } ( \pmb { x } ) , j \in S$ among all runs. The conservativeness metric is defined as the mean (with std) of the minimum value (in each run) of the safety specification $b _ { j } ( \pmb { x } ) , j \in S$ among all runs. The uncertainty metrics for both $u _ { 1 }$ and $u _ { 2 }$ are measured by the standard deviations of the model outputs (two controls) among all runs.


<table><tr><td>MODEL</td><td>CRASH (↓)</td><td>PASS (↑)</td><td>SAFETY (≥0)</td><td>CONSER. (≥0&amp;↓)</td><td><eq>u_1</eq> UNCER-TAINTY (↓)</td><td><eq>u_2</eq> UNCER-TAINTY (↓)</td><td>THEORET. GUAR.</td></tr><tr><td>V-E2E [2]</td><td>31%</td><td>69%</td><td>-59.455</td><td>-8.932±19.741</td><td>0.529</td><td>0.239</td><td>×</td></tr><tr><td>E2Es-MCD [12]</td><td>28%</td><td>72%</td><td>-58.405</td><td>-8.116±20.802</td><td>0.524</td><td>0.232</td><td>×</td></tr><tr><td>E2Es-DR [17]</td><td>27%</td><td>73%</td><td>-60.267</td><td>-8.781±20.910</td><td>0.512</td><td>0.225</td><td>×</td></tr><tr><td>DFB [23]</td><td>1%</td><td>37%</td><td>-13.281</td><td>-0.256±4.348</td><td>0.482</td><td>0.127</td><td>√</td></tr><tr><td>BNET [39]</td><td>23%</td><td>37%</td><td>-45.415</td><td>-9.114±13.382</td><td>0.730</td><td>0.316</td><td>√</td></tr><tr><td>BNET-UP [36]</td><td>24%</td><td>39%</td><td>-44.634</td><td>-8.866±13.167</td><td>0.747</td><td>0.278</td><td>×</td></tr><tr><td>ABNET (OURS)</td><td>0%</td><td>100%</td><td>4.268</td><td>8.315±2.147</td><td>0.151</td><td>0.326</td><td>√</td></tr><tr><td>ABNET-ATT (OURS)</td><td>0%</td><td>100%</td><td>5.986</td><td>7.032±0.405</td><td>0.118</td><td>0.213</td><td>√</td></tr><tr><td>ABNET-SC (OURS)</td><td>0%</td><td>100%</td><td>4.118</td><td>7.515±1.120</td><td>0.128</td><td>0.255</td><td>√</td></tr></table>
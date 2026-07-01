# BarrierNet: A Safety-Guaranteed Layer for Neural Networks

Wei Xiao 

WEIXY@MIT.EDU 

Ramin Hasani 

RHASANI@MIT.EDU 

Xiao Li 

XIAOLI@MIT.EDU 

Daniela Rus* 

RUS@MIT.EDU 

## Abstract

This paper introduces differentiable higher-order control barrier functions (CBF) that are end-toend trainable together with learning systems. CBFs are usually overly conservative, while guaranteeing safety. Here, we address their conservativeness by softening their definitions using environmental dependencies without loosing safety guarantees, and embed them into differentiable quadratic programs. These novel safety layers, termed a BarrierNet, can be used in conjunction with any neural network-based controller, and can be trained by gradient descent. BarrierNet allows the safety constraints of a neural controller be adaptable to changing environments. We evaluate them on a series of control problems such as traffic merging and robot navigations in 2D and 3D space, and demonstrate their effectiveness compared to state-of-the-art approaches. 

Keywords: Neural Network, Safety Guarantees, Control Barrier Function 

## 1. Introduction

The deployment of learning systems in decision-critical applications such as autonomous ground and aerial vehicle control is strictly conditional to their safety properties. This is because one simple mistake made by a learned controller, can lead to catastrophic outcomes. Notwithstanding, ensuring the safety (e.g., providing guarantees that a self-driving car will never collide with obstacles) of modern learning-based autonomous controllers is challenging. This is because these models perform representation learning end-to-end in unstructured environments, and are expected to generalize well to unseen situations. 

In this work, we take insights from designing data-driven safety controllers (Ames et al., 2014, 2017) to equip end-to-end learning systems with safety guarantees. To this end, we propose a fundamental algorithm to synthesize safe neural controllers end-to-end, by defining novel instances of control barrier functions (CBFs), that are differentiable. CBFs are popular methods for guaranteeing safety when the system dynamics are known. A large body of work studied variants of CBFs (Nguyen and Sreenath, 2016; Wang et al., 2018; Taylor et al., 2020a; Choi et al., 2020; Taylor et al., 2020b), and their characteristics under increasing uncertainty (Xu et al., 2015; Gurriet et al., 2018; Taylor and Ames, 2020). The common observation is that as the uncertainty of models increase CBFs make the system’s behavior excessively conservative (Csomay-Shanklin et al., 2021). 

In the present study, we address this over-conservativeness of CBFs by replacing the set of hard constraints in high order CBFs (HOCBFs) (Xiao and Belta, 2021) for arbitrary-relative-degree systems, with a set of differentiable soft constraints without loss of safety guarantees. This way, we obtain a versatile safety guaranteed barrier layer, termed a BarrierNet, that can be combined with any deep learning system (Hochreiter and Schmidhuber, 1997; He et al., 2016; Vaswani et al., 2017; Lechner and Hasani, 2020; Hasani et al., 2021a), and can be trained end-to-end via reverse-mode automatic differentiation (Rumelhart et al., 1986). 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/1b4253a9-0808-41d3-b3e9-787ae7aa33f1/86fe849f1e278130026c1199cdcebb3c34668f0cbb39f6e281f4051b93969291.jpg)



Figure 1: A safety guaranteed BarrierNet controller for autonomous driving. Collision avoidance is guaranteed. x is the vehicle state, and z is the observation variable. $\psi _ { i } ( { \pmb x } , z ) , i \in$ $\{ 1 , \ldots , m \}$ are a sequence of CBFs with their class $\kappa$ funtions $\alpha _ { i }$ in a HOCBF with relative degree m. $p _ { i } ( z ) , i \in \{ 1 , . . . , m \}$ are trainable parameters or from the previous layer. A standard HOCBF does not have the trainable terms $p _ { i } ( z )$ , and thus, each $\psi _ { i }$ is independent of the observation variable z.


BarrierNet allows the safety constraints of a neural controller to be adaptable to changing environments. A typical application of BarrierNet is autonomous driving that is shown in Fig. 1. In this example, a neural network outputs acceleration and steering commands to navigate the vehicle along the center lane while avoiding obstacles. The system and environmental observations are inputs to the upstream network whose outputs serve as arguments to the BarrierNet layer. Finally, BarrierNet outputs the controls that guarantee collision avoidance. In contrast to existing work, our proposed architecture is end-to-end trainable including the BarrierNet for neural networks. 

Contributions: (i) We propose a novel trainable and interpretable layer, built by leveraging the definition of higher order control barrier functions,that provides safety guarantees for general control problems with neural network controllers. (ii) We resolve the over-conservativeness of CBFs by introducing differentiable soft constraints in their definition (c.f., Fig. 1). This way we can train them end-to-end with a given learning system. (iii) We design BarrierNet such that the CBF parameters are adaptive to changes in environmental conditions. CBF parameters can be learned from data. We evaluate BarrierNet on a set of control problems including traffic merging and robot navigation with obstacles in 2D and 3D. 

This paper is organized as follows: In Sec. 2 and 3, we provide the related work and the necessary background to construct our theory, respectively. We formulate our problem in Sec. 4 and introduce BarrierNets in Sec. 5. Sec. 6 includes our experimental evaluation and we conclude the paper in Sec. 7. 

## 2. Related Work

Control Barrier Function and Learning. A large body of work studied CBF-based safety guarantees (Nguyen and Sreenath, 2016; Wang et al., 2018; Taylor et al., 2020a; Choi et al., 2020; Taylor et al., 2020b), and their characteristics under increasing model uncertainty (Xu et al., 2015; Gurriet et al., 2018; Taylor and Ames, 2020). Many existing works (Ames et al., 2017; Nguyen and Sreenath, 2016; Yang et al., 2019) combine CBFs for systems with quadratic costs to form optimization problems. Time is discretized and an optimization problem with constraints given by the CBFs (inequalities of the form (4 )) which are solved at each time step. The inter-sampling effect is considered in (Yang et al., 2019; Xiao et al., 2021a). Replacing CBFs by HOCBFs allows us to handle constraints with arbitrary relative degree (Xiao and Belta, 2021). The common observation is that as the uncertainty of models increases, CBFs make the system’s behavior excessively conservative (Csomay-Shanklin et al., 2021). 

The recently proposed adaptive CBFs (AdaCBFs) (Xiao et al., 2021b) addressed the conservativeness of the HOCBF method by multiplying the class K functions of an HOCBF with some penalty functions. These penalties functions, themselves, are HOCBFs such that they are guaranteed to be non-negative. This is due to the fact that the main conservativeness of the HOCBF method comes from the class K functions. By multiplying (relaxing) the class K functions with some penalty functions, it has been shown that the satisfaction of the AdaCBF constraint is a necessary and sufficient condition for the satisfaction of the original safety constraint $b ( { \pmb x } ) \geq 0$ (Xiao et al., 2021b). This is conditioned on designing proper auxiliary dynamics for all the penalty functions, based on specific problems. However, how to design such auxiliary dynamics is still a remaining challenge, which we study here. 

At the intersection of CBFs and learning, supervised learning techniques have been proposed to learn safe set definitions from demonstrations (Robey et al., 2020), and sensor data (Srinivasan et al., 2020) which are then enforced by CBFs. (Taylor et al., 2020a) used data to learn system dynamics for CBFs. In a similar setting, (Lopez et al., 2020) used adaptive control approaches to estimate the unknown system parameters in CBFs. In (Yaghoubi et al., 2020), neural network controllers are trained using CBFs in the presence of disturbances. These prior works focus on learning safe sets and dynamics, whereas we focus on the design of environment dependent and trainable CBFs. 

Optimization-based safety frameworks. Recent advances in differentiable optimization methods show promising directions for safety guaranteed neural network controllers (Lechner et al., 2020b; Gruenbacher et al., 2020; Grunbacher et al., 2021; Gruenbacher et al., 2021; Massiani et al., 2021; Lechner et al., 2021). In (Amos and Kolter, 2017), a differentiable quadratic program (QP) layer, called OptNet, was introduced. The OptNet has been used in neural networks as a filter for safe controls (Pereira et al., 2020), in which case the OptNet is not trainable, thus, could limit the system’s learning performance. In contrast, we propose a trainable safety-guaranteed neural controller. In (Deshmukh et al., 2019; Jin et al., 2020; Zhao et al., 2021), safety guaranteed neural network controllers have been learned through verification-in-the-loop training. A safe neural network filter has been proposed in (Ferlez et al., 2020) for a specific vehicle model using verification methods. The verification approaches can not ensure coverage of the entire state space, they are offline and are not adaptive to environment changes (such as varying size of the unsafe sets) (Li, 2021). In comparison, BarrierNet is online, easily scalable, general to control problems and is adaptive to the environment changes. 

## 3. Backgrounds

In this section, we briefly introduce control barrier functions (CBF) and refer interested readers to (Ames et al., 2017) for detailed formulations. Intuitively, CBF is a means to translate state constraints to control constraints under affine dynamics. The controls that satisfy those constraints can be efficiently solved for by formulating a quadratic program. We start with the definition of a class K function. 

Definition 1 (Class K function (Khalil, 2002)) A continuous function α : $[ 0 , a ) \to [ 0 , \infty ) , a > 0$ is said to belong to class K if it is strictly increasing and $\alpha ( 0 ) = 0 .$ . A continuous function $\beta : \mathbb { R } $ R is said to belong to extended class K if it is strictly increasing and $\beta ( 0 ) = 0$ . 

Consider an affine control system of the form 

$$
\dot {\boldsymbol {x}} = f (\boldsymbol {x}) + g (\boldsymbol {x}) \boldsymbol {u} \tag {1}
$$

where $\pmb { x } \in \mathbb { R } ^ { n } , f : \mathbb { R } ^ { n }  \mathbb { R } ^ { n }$ and $g : \mathbb { R } ^ { n }  \mathbb { R } ^ { n \times q }$ are locally Lipschitz, and $\pmb { u } \in U \subset \mathbb { R } ^ { q }$ , where U denotes a control constraint set. 

Definition 2 A set $C \subset \mathbb { R } ^ { n }$ is forward invariant for system (1) if its solutions for some $\textbf { \em u } \in \mathbf { \Xi } U$ starting at any $\pmb { x } ( 0 ) \in C$ satisfy ${ \pmb x } ( t ) \in C , \forall t \geq 0 .$ . 

Definition 3 (Relative degree) The relative degree of a (sufficiently many times) differentiable function $b : \mathbb { R } ^ { n } $ R with respect to system (1) is the number of times it needs to be differentiated along its dynamics until the control u explicitly shows in the corresponding derivative. 

Since function b is used to define a (safety) constraint $b ( { \pmb x } ) \geq 0$ , we will also refer to the relative degree of b as the relative degree of the constraint. For a constraint $b ( { \pmb x } ) \geq 0$ with relative degree m, $b : \mathbb { R } ^ { n }  \mathbb { R }$ , and $\psi _ { 0 } ( { \pmb x } ) : = b ( { \pmb x } )$ , we define a sequence of functions $\psi _ { i } : \mathbb { R } ^ { n }  \mathbb { R } , i \in \{ 1 , \dots , m \}$ : 

$$
\psi_ {i} (\boldsymbol {x}) := \dot {\psi} _ {i - 1} (\boldsymbol {x}) + \alpha_ {i} \left(\psi_ {i - 1} (\boldsymbol {x})\right), \quad i \in \{1, \dots , m \}, \tag {2}
$$

where $\alpha _ { i } ( \cdot ) , i \in \{ 1 , \ldots , m \}$ denotes a $( m - i ) ^ { t h }$ order differentiable class K function. 

We further define a sequence of sets $C _ { i } , i \in \{ 1 , . . . , m \}$ associated with (2) in the form: 

$$
C _ {i} := \{\boldsymbol {x} \in \mathbb {R} ^ {n}: \psi_ {i - 1} (\boldsymbol {x}) \geq 0 \}, \quad i \in \{1, \dots , m \}. \tag {3}
$$

Definition 4 (High Order Control Barrier Function (HOCBF) (Xiao and Belta, 2021)) Let $C _ { 1 } , \ldots , C _ { m }$ be defined by (3) and $\psi _ { 1 } ( \pmb { x } ) , \ldots , \psi _ { m } ( \pmb { x } )$ be defined by (2). A function $b : \mathbb { R } ^ { n }  \mathbb { R }$ is a High Order Control Barrier Function (HOCBF) of relative degree m for system (1) if there exist $( m - i ) ^ { t h }$ order differentiable class K functions $\alpha _ { i } , i \in \{ 1 , \ldots , m - 1 \}$ and a class K function $\alpha _ { m }$ such that 

$$
\sup _ {\boldsymbol {u} \in U} [ L _ {f} ^ {m} b (\boldsymbol {x}) + [ L _ {g} L _ {f} ^ {m - 1} b (\boldsymbol {x}) ] \boldsymbol {u} + O (b (\boldsymbol {x})) + \alpha_ {m} (\psi_ {m - 1} (\boldsymbol {x})) ] \geq 0, \tag {4}
$$

for all $\pmb { x } \in C _ { 1 } \cap , . . . , \cap C _ { m }$ . In (4), Lmf $( L _ { g } )$ denotes Lie derivatives along $f \left( g \right)$ m (one) times, and $\begin{array} { r } { O ( b ( \pmb { x } ) ) = \sum _ { i = 1 } ^ { m - 1 } L _ { f } ^ { i } ( \alpha _ { m - i } \circ \psi _ { m - i - 1 } ) ( \pmb { x } ) } \end{array}$ $b ( { \pmb x } )$ $L _ { g } L _ { f } ^ { m - 1 } b ( { \pmb x } ) \neq 0$ boundary of the set $\ddot { C _ { 1 } } \cap , \dotsc , \cap C _ { m }$ . 


Table 1: Notation


<table><tr><td>Symbol</td><td>Definition</td></tr><tr><td>t</td><td>time</td></tr><tr><td>θ</td><td>BarrierNet parameters</td></tr><tr><td><eq>\boldsymbol{x} \in \mathbb{R}^{n}</eq></td><td>vehicle state</td></tr><tr><td><eq>\boldsymbol{z} \in \mathbb{R}^{d}</eq></td><td>observation variable</td></tr><tr><td><eq>\boldsymbol{u} \in \mathbb{R}^{q}</eq></td><td>control</td></tr><tr><td><eq>\alpha : [0, a) \to [0, \infty), a &gt; 0</eq></td><td>class <eq>\mathcal{K}</eq> function</td></tr><tr><td><eq>\beta : \mathbb{R} \to \mathbb{R}</eq></td><td>extended class <eq>\mathcal{K}</eq> function</td></tr><tr><td><eq>b : \mathbb{R}^{n} \to \mathbb{R}</eq></td><td>safety constraint</td></tr><tr><td><eq>\psi : \mathbb{R}^{n} \to \mathbb{R}</eq></td><td>CBF</td></tr><tr><td><eq>p : \mathbb{R}^{d} \to \mathbb{R}^{&gt;0}</eq></td><td>penalty function</td></tr><tr><td><eq>C \subset \mathbb{R}^{n}</eq></td><td>safe set</td></tr><tr><td><eq>l : \mathbb{R}^{q} \times \mathbb{R}^{q} \to \mathbb{R}</eq></td><td>similarity measure</td></tr><tr><td><eq>f : \mathbb{R}^{n} \to \mathbb{R}^{n}</eq></td><td>affine dynamics drift term</td></tr><tr><td><eq>g : \mathbb{R}^{n} \to \mathbb{R}^{n \times q}</eq></td><td>affine dynamics control term</td></tr></table>

The HOCBF is a general form of the relative degree one CBF (Ames et al., 2017) (setting $m = 1$ reduces the HOCBF to the common CBF form). Note that we can define $\alpha _ { i } ( \cdot ) , i \in \{ 1 , \ldots , m \}$ in Def. 4 to be extended class K functions to ensure robustness of a HOCBF to perturbations (Ames et al., 2017). However, the use of extended class K functions cannot ensure a constraint is eventually satisfied if it is initially violated. 

**Theorem 5** (*Xiao and Belta, 2021*) Given a HOCBF $ b(\boldsymbol{x}) $ from Def. 4 with the associated sets $ C_1, \dots, C_m $ defined by (3), if $ \boldsymbol{x}(0) \in C_1 \cap \dots \cap C_m $, then any Lipschitz continuous controller $ \boldsymbol{u}(t) $ that satisfies the constraint in (4), $ \forall t \geq 0 $ renders $ C_1 \cap \dots \cap C_m $ forward invariant for system (1).

We provide a summary of notations in Table 1. 

## 4. Problem Formulation

Here, we formally define the learning problem for safety-critical control. 

Problem 1 Given (i) a system with known affine dynamics in the form of 1, (ii) a nominal controller $f ^ { \star } ( x ) = u ^ { \star }$ , (iii) a set of safety constraints $b _ { j } ( \pmb { x } ) \geq 0 , j \in S$ (where $b _ { j }$ is continuously differentiable), (iv) control bounds ${ \pmb u } _ { m i n } \le { \pmb u } \le { \pmb u } _ { m a x }$ and (v) a neural network controller $f ( \mathbf { \mathscr { x } } | \boldsymbol { \theta } ) = \mathbf { \mathscr { u } }$ parameterized by θ, our goal is to find the optimal parameters 

$$
\theta^ {\star} = \underset {\theta} {\arg \min} \mathbb {E} _ {\boldsymbol {x}} [ l (f ^ {\star} (\boldsymbol {x}), f (\boldsymbol {x} | \theta)) ] \tag {5}
$$

while guaranteeing the satisfaction of the safety constraints in (iii) and control bounds in $( i \nu ) . \mathbb { E } ( \cdot )$ is the expectation and $l ( \cdot , \cdot )$ denotes a similarity measure. 

Problem 1 defines a policy distillation problem with safety guarantees. The safety constraints can be pre-defined by users or can be learned from data (Robey et al., 2020), (Srinivasan et al., 2020). 

## 5. BarrierNet

In this section, we propose BarrierNet - an CBF-based neural network controller with parameters that are trainable via backpropagation. We define the safety guarantees of a neural network controller as follows: 

Definition 6 (Safety guarantees) A neural network controller has safety guarantees for system (1) if its outputs (controls) drive system (1) such that $b _ { j } ( { \pmb x } ( t ) ) \geq 0 , \forall t \geq 0 , \forall j \in S ,$ , for continuously differentiable $b _ { j } : \mathbb { R } ^ { n }  \mathbb { R }$ . 

We start with the motivations of a BarrierNet. The HOCBF method provides safety guarantees for control systems with arbitrary relative degrees, but in a conservative way. In other words, the satisfaction of the HOCBF constraint (4) is only a sufficient condition of the satisfaction of the original safety constraint $b ( { \pmb x } ) \geq 0$ . This conservativeness of the HOCBF method will significantly limit the system performance. For example, such conservativeness may drive the system much further away from obstacles than necessary. Our first motivation is to address this conservativeness. 

More specifically, a HOCBF constraint is always a hard constraint in order to guarantee safety. This may adversely affect the performance of the system. In order to address this, we soften a HOCBF in a way without loosing the safety guarantees. In addition, we also incorporate HOCBF in a differentiable optimization layer to allow tuning of its parameters from data. 

Given a safety requirement $b ( { \pmb x } ) \geq 0$ with relative degree m for system (1), we change the sequence of CBFs in (2) by: 

$$
\psi_ {i} (\boldsymbol {x}, \boldsymbol {z}) := \dot {\psi} _ {i - 1} (\boldsymbol {x}, \boldsymbol {z}) + p _ {i} (\boldsymbol {z}) \alpha_ {i} \left(\psi_ {i - 1} (\boldsymbol {x}, \boldsymbol {z})\right), \quad i \in \{1, \dots , m \}, \tag {6}
$$

where $\psi _ { 0 } ( { \pmb x } , z ) = b ( { \pmb x } )$ and $z \in \mathbb { R } ^ { d }$ are the input of the neural network $( d \in \mathbb { N }$ is the dimension of the features), $p _ { i } : \mathbb { R } ^ { d }  \mathbb { R } ^ { > 0 } , i \in \{ 1 , \dots , m \}$ are the outputs of the previous layer, where $\mathbb { R } { > } 0$ denotes the set of positive scalars. The above formulation is similar to the Adaptive CBF (AdaCBF) (Xiao et al., 2021b), but is trainable, and does not require us to design auxiliary dynamics for $p _ { i }$ (an AdaCBF does require), a non-trivial process of the existing AdaCBF method. In order to make sure that the HOCBF method can be solved in a time-discretization way (Ames et al., 2017), we require that each $p _ { i }$ is Lipschitz continuous. Then, we have a similar HOCBF constraint as in Def. 4 in the form: 

$$
L _ {f} ^ {m} b (\boldsymbol {x}) + \left[ L _ {g} L _ {f} ^ {m - 1} b (\boldsymbol {x}) \right] \boldsymbol {u} + O (b (\boldsymbol {x}), \boldsymbol {z}) + p _ {m} (\boldsymbol {z}) \alpha_ {m} \left(\psi_ {m - 1} (\boldsymbol {x}, \boldsymbol {z})\right) \geq 0, \tag {7}
$$

Following the discretization solving method introduced in (Ames et al., 2017), if we wish to minimize the control input effort (in the case where the reference control is 0), we can reformulate our problem as the following sequence of QPs: 

$$
\boldsymbol {u} ^ {*} (t) = \arg \min _ {\boldsymbol {u} (t)} \frac {1}{2} \boldsymbol {u} (t) ^ {T} H \boldsymbol {u} (t) \tag {8}
$$

s.t. 

$$
L _ {f} ^ {m} b _ {j} (\boldsymbol {x}) + \left[ L _ {g} L _ {f} ^ {m - 1} b _ {j} (\boldsymbol {x}) \right] \boldsymbol {u} + O (b _ {j} (\boldsymbol {x}), \boldsymbol {z}) + p _ {m} (\boldsymbol {z}) \alpha_ {m} \left(\psi_ {m - 1} (\boldsymbol {x}, \boldsymbol {z})\right) \geq 0, j \in S
$$

$$
\boldsymbol {u} _ {\text { min }} \leq \boldsymbol {u} \leq \boldsymbol {u} _ {\text { max }},
$$

$$
t = k \Delta t + t _ {0},
$$

where $\Delta t > 0$ is the discretization time. Since the QPs in (8) is solved pointwise, the resulting solutions would be sub-optimal. In order to address this problem and be able to track any nominal controllers, we introduce BarrierNet. 

Definition 7 (BarrierNet) A BarrierNet is composed by neurons in the form: 

$$
\boldsymbol {u} ^ {*} (t) = \arg \min _ {\boldsymbol {u} (t)} \frac {1}{2} \boldsymbol {u} (t) ^ {T} H (\boldsymbol {z} | \theta_ {h}) \boldsymbol {u} (t) + F ^ {T} (\boldsymbol {z} | \theta_ {f}) \boldsymbol {u} (t) \tag {9}
$$

s.t. 

$$
L _ {f} ^ {m} b _ {j} (\boldsymbol {x}) + \left[ L _ {g} L _ {f} ^ {m - 1} b _ {j} (\boldsymbol {x}) \right] \boldsymbol {u} + O \left(b _ {j} (\boldsymbol {x}), \boldsymbol {z} \mid \theta_ {p}\right) + p _ {m} \left(\boldsymbol {z} \mid \theta_ {p} ^ {m}\right) \alpha_ {m} \left(\psi_ {m - 1} (\boldsymbol {x}, \boldsymbol {z} \mid \theta_ {p})\right) \geq 0, j \in S \tag {10}
$$

$$
\boldsymbol {u} _ {\text { min }} \leq \boldsymbol {u} \leq \boldsymbol {u} _ {\text { max }},
$$

$$
t = k \Delta t + t _ {0},
$$

where $F ( z | \theta _ { f } ) \in \mathbb { R } ^ { q }$ could be interpreted as a reference control (can be the output of previous network layers) and $\theta _ { h } , \theta _ { f } , \theta _ { p } = ( \theta _ { p } ^ { 1 } , \ldots , \theta _ { p } ^ { m } )$ are trainable parameters. 

The inequality (10) in Definition 7 guarantees each safety constraint $b _ { j } ( { \pmb x } ) \geq 0 , \forall j \in S$ through the parameterized function $p _ { i } , i \in \{ 1 , \ldots , m \}$ . Based on Def. 7, for instance, if we have 10 control agents, we need 10 BarrierNet neurons presented by (9) to ensure the safety of each agent. This implies that BarrierNets can be extended to multi-agent settings. 

In Def. 7, we make both $H ( z | \theta _ { h } )$ and $F ( z | \theta _ { f } )$ parameterized and dependent on the network input z, but H and F can also be directly trainable parameters that do not depend on the previous layer (i.e., we have H and F ). The same applies to $p _ { i } , i \in \{ 1 , \ldots , m \}$ . The trainable parameters are $\theta = \{ \theta _ { h } , \theta _ { f } , \theta _ { p } \} ( \mathrm { o r } \theta = \{ H , F , p _ { i } , \forall i \in \{ 1 , \dots , m \} \}$ } if H, F and $p _ { i }$ do not depend on the previous layer). The solution $\pmb { u } ^ { * }$ is the output of the neuron. The BarrierNet is differentiable with respect to its parameters (Amos and Kolter, 2017). We describe the forward and backward passes as follows. 

Forward pass: The forward step of a BarrierNet is to solve the QP in Definition 7. The inputs of a BarrierNet include environmental features z (such as the location and speed of an obstacle) that can be provided directly or from a tracking network if raw sensory inputs are used. BarrierNet also takes as inputs the system states x as a feedback, as shown in Fig. 1. The outputs are the solutions of the QP (the resultant controls). 

Backward pass: The main task of BarrierNet is to provide controls while always ensuring safety. Suppose ` denotes some loss function (a similarity measure). Using the techniques introduced in (Amos and Kolter, 2017), the relevant gradient with respect to all the BarrierNet parameters can be given by (the gradient with respect to the parameters $\theta _ { h } , \theta _ { f } , \theta _ { p }$ can be obtained using the chain rule): 

$$
\bigtriangledown_ {H} \ell = \frac {1}{2} (d _ {\boldsymbol {u}} \boldsymbol {u} ^ {T} + \boldsymbol {u} d _ {\boldsymbol {u}} ^ {T}), \quad \bigtriangledown_ {F} \ell = d _ {\boldsymbol {u}}, \tag {11}
$$

$$
\bigtriangledown_ {G} \ell = D (\lambda^ {*}) (d _ {\lambda} \pmb {u} ^ {T} + \lambda d _ {\pmb {u}} ^ {T}), \bigtriangledown_ {h} \ell = - D (\lambda^ {*}) d _ {\lambda},
$$

where λ are the dual variables on the HOCBF constraints and $D ( \cdot )$ creates diagonal matrix from a vector. $G , h$ are concatenated by $G _ { j } , h _ { j } , j \in S$ , where 

$$
G _ {j} = - L _ {g} L _ {f} ^ {m - 1} b _ {j} (\boldsymbol {x}), \tag {12}
$$

$$
h _ {j} = L _ {f} ^ {m} b _ {j} (\boldsymbol {x}) + O (b _ {j} (\boldsymbol {x}), \boldsymbol {z}) + p _ {m} (\boldsymbol {z}) \alpha_ {m} (\psi_ {m - 1} (\boldsymbol {x}, \boldsymbol {z})).
$$

Since the control bounds in (9) are not trainable, they are not included in $G , h .$ . 

In (11), $d _ { u }$ and $d _ { \lambda }$ are given by solving: 

$$
\left[ \begin{array}{l} d _ {\boldsymbol {u}} \\ d _ {\lambda} \end{array} \right] = \left[ \begin{array}{c c} H & G ^ {T} D (\lambda^ {*}) \\ G & D (G \boldsymbol {u} ^ {*} - h) \end{array} \right] ^ {- 1} \left[ \begin{array}{c} (\frac {\partial \ell}{\partial \boldsymbol {u} ^ {*}}) ^ {T} \\ 0 \end{array} \right], \tag {13}
$$

The above equation is obtained by taking the Lagrangian of QP followed by applying the Karush–Kuhn–Tucker conditions. $\nabla G ^ { \ell }$ is not applicable in a BarrierNet as it is determined by the corresponding HOCBF. $\nabla h ^ { \ell }$ is also not directly related to the input of a BarrierNet. Nevertheless, we have $\nabla _ { p _ { i } } \ell = \nabla _ { h _ { i } } \ell _ { \mathbf { \tau } } \nabla _ { p _ { i } } h _ { j } , i \in \{ 1 , \dots , m \} , j \in S$ ,where $\nabla h _ { j } \ell$ is given by $\nabla h ^ { \ell }$ in (11) and $\nabla _ { p _ { i } } h _ { j }$ is given by taking the partial derivative of $h _ { j }$ in (12). 

The Following theorem characterizes the safety guarantees of a BarrierNet: 

Theorem 8 $I f p _ { i } ( z ) , i \in \{ 1 , \ldots , m \}$ are Lipschitz continuous, then a BarrerNet composed by neurons as in Def. 7 guarantees the safety of system (1). 

proof: Since $p _ { i } ( z ) , i \in \{ 1 , . . . , m \}$ are Lipschitz continuous, it follows from Thm. 5 that each $\psi _ { i } ( { \pmb x } , z )$ in (6) is a valid CBF. Starting from $\psi _ { m } ( \pmb { x } , z ) \geq 0$ (the non-Lie-derivative form of each HOCBF constraint (7)), we can show that $\psi _ { m - 1 } ( \pmb { x } , z ) \geq 0$ is guaranteed to be satisfied following Thm. 5. Recursively, we can show that $\psi _ { 0 } ( { \pmb x } , z ) \ge 0$ is guaranteed to be satisfied. As $b ( { \pmb x } ) =$ $\psi _ { 0 } ( { \pmb x } , z )$ following (6), we have that system (1) is safety guaranteed in a BarrierNet.  

Remark 9 (Adaptivity of the BarrierNet) The HOCBF constraints in a BarrierNet are softened by the trainable penalty functions without loosing safety guarantees. The penalty functions are environment dependent which features can be calculated from upstream networks. The adaptive property of the HOCBFs provides the adaptivity of the BarrierNet. As a result, BarrierNet is able to generate safe controls while avoid being overly conservative. 

In order to guarantee that $p _ { i } ( z ) , i \in \{ 1 , . . . , m \}$ are Lipschitz continuous, we can choose activation functions of the previous layer as some continuously differentiable functions, such as sigmoid functions. The process of constructing and training a BarrierNet includes: (a) construct a softened HOCBF by (6) that enforces each of the safety requirement, (b) construct the parameterized BarrierNet by (9), (c) get the training data set using the nominal controller, and (d) train the BarrierNet using error backpropogation. We summarize the algorithm for the BarrierNet in Alg. 1. 

Limitations: The proposed BarrierNet can theoretically guarantee system safety. However, there are also some limitations: 

(i) The number of safety constraints in a BarrierNet should be defined in training. In some scenarios, the number of safety constraints may be time-varying. Therefore, how to match multiple safety constraints with the definition of a BarrierNet remains a challenge. It is possible that we may define more than necessary constraints in a BarrierNet, and only enable those when required. 

(ii) The BarrierNet is solved in discrete time, as we can only feed discrete data into the neural network. The inter-sampling effect (the system’s trajectory between time intervals) should also be considered in order to achieve safety guarantees. The inter-sampling effect is sensitive at the boundary of the safety set. Therefore, we need to avoid sampling training data at the safety set boundary. However, due to the Lyapunov property of HOCBFs, the system will always stay close to the safety set boundary if the safety constraint is violated due to the inter-sampling effect, or some perturbations. A possible approach to address the inter-sampling effect is the data-driven event-triggered framework (Xiao et al., 2021a). 


Algorithm 1 Construction and training of the BarrierNet


Input: Dynamics (1), Safety requirements in Problem 1, a nominal controller
Output: A safety-guaranteed BarrierNet controller.
(a) Construct softened HOCBFs by (6)
(b) Construct the BarrierNet by (9)
(c) Get the training data set using the nominal controller
(d) Initialize the BarrierNet parameter $\theta$ , the Epochs, and the learning rate $\gamma$ while $e$ in Epochs do
    Forward: Solve (9) and get the loss $\ell$ Backward: Get the loss gradient $\bigtriangledown_{\theta}\ell$ using (11) $\theta \leftarrow \theta - \gamma \bigtriangledown_{\theta} \ell$ end
return $\theta$ (optimal parameters in the BarrierNet) 

## 6. Numerical Evaluations

In this section, we present three case studies (a traffic merging control problem and robot navigation problems in 2D and 3D) to verify the effectiveness of our proposed BarrierNet. 

## 6.1. Traffic Merging Control

Experiment setup. The traffic merging problem arises when traffic must be joined from two different roads, usually associated with a main lane and a merging lane as shown in Fig.1. We consider the case where all traffic consisting of controlled autonomous vehicles (CAVs) arrive randomly at the origin (O and $O ^ { \prime } )$ and join at the Merging Point (MP) M where a lateral collision may occur. The segment from the origin to the merging point M has length L for both lanes, and is called the Control Zone (CZ). All CAVs do not overtake each other in the CZ as each road consists of a single lane. A coordinator is associated with the MP whose function is to maintain a First-In-First-Out (FIFO) queue of CAVs based on their arrival time at the CZ. The coordinator also enables real-time communication among the CAVs that are in the CZ including the last one leaving the CZ. The FIFO assumption, imposed so that CAVs cross the MP in their order of arrival, is made for simplicity and often to ensure fairness. 

Notation. $x _ { k } , v _ { k } , u _ { k }$ denote the along-lane position, speed and acceleration (control) of CAV k, re-$t _ { k } ^ { 0 } , t _ { k } ^ { m }$ $\boldsymbol { z } _ { k , k _ { p } }$ denotes the along lane distance between CAV k and its preceding CAV $k _ { p } ,$ as shown in Fig. 2. 

Our goal is to jointly minimize all vehicles’ travel time and energy consumption in the control zone. Written as an objective function, we have 

$$
\min _ {u _ {k} (t)} \beta (t _ {k} ^ {m} - t _ {k} ^ {0}) + \int_ {t _ {k} ^ {0}} ^ {t _ {k} ^ {m}} \frac {1}{2} u _ {k} ^ {2} (t) d t, \tag {14}
$$

where $u _ { k }$ is the vehicle’s control (acceleration), and $\beta > 0$ is a weight controlling the relative magnitude of travel time and energy consumption. We assume double integrator dynamics for all vehicles. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/1b4253a9-0808-41d3-b3e9-787ae7aa33f1/fd1defcc52b86edf215cd291efea444e35d91d83d91af0c91dc3d3e67a29e7a8.jpg)



Figure 2: A traffic merging problem. A collision may happen at the merging point as well as everywhere within the control zone.


Each vehicle k should satisfy the following rear-end safety constraint if its preceding vehicle $k _ { p }$ is in the same lane: 

$$
z _ {k, k _ {p}} (t) \geq \phi v _ {k} (t) + \delta , \forall t \in [ t _ {k} ^ {0}, t _ {k} ^ {m} ], \tag {15}
$$

where $z _ { k , k _ { p } } = x _ { k _ { p } } - x _ { k }$ denotes the along-lane distance between k and $k _ { p } , \phi$ is the reaction time (usually takes 1.8s) and $\delta \geq 0$ . 

The traffic merging problem is to find an optimal control that minimizes (14), subject to (15). We assume vehicle k has access to only the information of its immediate neighbors from the coordinator (shown in Fig. 2), such as the preceding vehicle $k _ { p }$ . This merging problem can be solved analytically by optimal control methods (Xiao and Cassandras, 2021), but at the cost of extensive computation, and the solution becomes complicated when one or more constraints become active in an optimal trajectory, hence possibly prohibitive for real-time implementation. 

BarrierNet design. We enforce the safety constraint (15) by a CBF $b ( z _ { k , k _ { p } } , v _ { k } ) = z _ { k , k _ { p } } ( t ) -$ $\phi v _ { k } ( t ) - \delta$ , and any control input $u _ { k }$ should satisfy the CBF constraint (4) which in this case (choose $\alpha _ { 1 }$ as a linear function in Def. 4) is : 

$$
\varphi u _ {k} (t) \leq v _ {k} (t) - v _ {k _ {p}} (t) + p _ {1} (\boldsymbol {z}) \left(z _ {k, k _ {p}} (t) - \phi v _ {k} (t) - \delta\right) \tag {16}
$$

where $v _ { k }$ is the speed of vehicle k and ${ \boldsymbol { z } } = ( x _ { k _ { p } } , v _ { k _ { p } } , x _ { k } , v _ { k } )$ is the input of the neural network model (to be designed later). $p _ { 1 } ( z )$ is called a penalty in the CBF that addresses the conservativeness of the CBF method. The cost in the neuron of the BarrierNet is given by: 

$$
\min _ {u _ {k}} \left(u _ {k} - f _ {1} (\boldsymbol {z})\right) ^ {2} \tag {17}
$$

where $f _ { 1 } ( z )$ is a reference to be trained (the output of the FC network). Then, we create a neural network model whose structure is composed by a fully connected (FC) network (an input layer and two hidden layers) followed by a BarrierNet. The input of the FC network is $z ,$ and its output is the penalty $p _ { 1 } ( z )$ and the reference $f _ { 1 } ( z )$ . While the input of the BarrierNet is the penalty $p _ { 1 } ( z )$ and the reference $f _ { 1 } ( z )$ , and its output is applied to control a vehicle k in the control zone. 

Results and discussion. To get the training data set, we solve an optimal or joint optimal control and barrier function (OCBF) (Xiao et al., 2021c) controller offline. The solutions of an optimal or 

OCBF controller are taken as labels. The training results with the optimal controller and the OCBF controller are shown in Figs. 3 - 5. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/1b4253a9-0808-41d3-b3e9-787ae7aa33f1/82722b741622f36d3117a7169cd0594896054e95e3c836eac039ae8bc9097117.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/1b4253a9-0808-41d3-b3e9-787ae7aa33f1/da078a11c86b3bee4531a140c1e56660b5cd2ad1cc94a6de0c3d1f116d5c5aba.jpg)



Figure 3: The control and penalty function $p _ { 1 } ( z )$ from the BarrierNet when training with the optimal controller. The blues curves (labeled as implementation) are the vehicle control when we apply the BarrierNet to drive the vehicle dynamics to pass through the control zone.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/1b4253a9-0808-41d3-b3e9-787ae7aa33f1/b3f0b0473300007321d265659bf7e009a86a8882096815642926ca7044f79765.jpg)



(a) Training using the optimal controller.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/1b4253a9-0808-41d3-b3e9-787ae7aa33f1/7cd3025bbf40ad39b0d1d2b1371fac59ec00874dd8766b1e7b7c18f1728b9371.jpg)



(b) Training using the OCBF controller



Figure 4: The safety comparison (under 10 trained models) between the BarrierNet and a FC network when training using the optimal/OCBF controller $( \delta \ : = \ : 0 )$ . If $z _ { k , k _ { p } } / v _ { k }$ is above the line $\phi = 1 . 8 ,$ , then safety is guaranteed. We observe that only neural network agents equipped with BarrierNet satisfy this condition.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/1b4253a9-0808-41d3-b3e9-787ae7aa33f1/815284d3e43e74a822cfddd3f13ef9be2e7fcd2c4159579d84149491f9268ce1.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/1b4253a9-0808-41d3-b3e9-787ae7aa33f1/34902c948906a2515759fc80a68414f275f76daaf61b8d4c074ee68773097f3c.jpg)



Figure 5: The control and penalty function $p _ { 1 } ( z )$ from the BarrierNet when training with the OCBF controller. The blues curves (labeled as implementation) are the vehicle control when we apply the BarrierNet to drive the vehicle dynamics to pass through the control zone.


In an optimal controller, the original safety constraint is active after around $6 s ,$ as shown in Fig. 3. Therefore, the sampling trajectory is on the safety boundary, and inter-sampling effect becomes important in this case. Since we do not consider the inter-sampling effect in this paper, the safety metric of the BarrierNet might go below the lower bound $\phi = 1 . 8$ , as the red curves shown in Fig. $4 \mathrm { a } .$ However, due to the Lyapunov property of the CBF, the safety metric will always stay close to the lower bound $\phi = 1 . 8$ . The solutions for 10 trained models are also consistent. In a FC network, the safety metrics vary under different trained models, and the safety constraint might be violated, as the blue curves shown in Fig. 4a. 

In an OCBF controller, the original safety constraint is not active, and thus, the inter-sampling effect is not sensitive. As shown in Figs. 4b, safety is always guaranteed in a BarrierNet under 10 trained models. While in a FC network, the safety constraint may be violated as there are no guarantees. 

We present the penalty functions when training with the optimal controller and the OCBF controller in Figs. 3 and 5, respectively. The penalty function $p _ { 1 } ( z )$ decreases when the CBF constraint becomes active. This shows the adaptivity of the BarrierNet. This behavior is similar to the AdaCBF, but in the BarrierNet, we do not need to design auxiliary dynamics for the penalty functions. Therefore, the BarrierNet is simpler than the AdaCBF. Finally, we present a comprehensive comparison between the BarrierNet, the FC network, the optimal controller and the OCBF controller in Table 2. 


Table 2: Comparisons between the BarrierNet, the FC network, the optimal controller and the OCBF controller


<table><tr><td>item</td><td>R.T. compute time</td><td>safety guarantee</td><td>Optimality</td><td>Adaptive</td></tr><tr><td>BarrierNet</td><td>&lt; 0.01s</td><td>Yes</td><td>close-optimal</td><td>Yes</td></tr><tr><td>FC</td><td>&lt; 0.01s</td><td>No</td><td>close-optimal</td><td>No</td></tr><tr><td>Optimal</td><td>30s</td><td>Yes</td><td>optimal</td><td>Yes</td></tr><tr><td>OCBF</td><td>&lt; 0.01s</td><td>Yes</td><td>sub-optimal</td><td>Yes</td></tr></table>

## 6.2. 2D Robot Navigation

Experiment setup. We consider a robot navigation problem with obstacle avoidance. In this case, we consider nonlinear dynamics with two control inputs and nonlinear safety constraints. The robot navigates according to the following unicycle model for a wheeled mobile robot: 

$$
\left[ \begin{array}{c} \dot {x} \\ \dot {y} \\ \dot {\theta} \\ \dot {v} \end{array} \right] = \left[ \begin{array}{c} v \cos (\theta) \\ v \sin (\theta) \\ 0 \\ 0 \end{array} \right] + \left[ \begin{array}{c c} 0 & 0 \\ 0 & 0 \\ 1 & 0 \\ 0 & 1 \end{array} \right] \left[ \begin{array}{c} u _ {1} \\ u _ {2} \end{array} \right] \tag {18}
$$

where $\pmb { x } : = ( x , y , \theta , v ) , \pmb { u } = ( u _ { 1 } , u _ { 2 } ) , x$ , y denote the robot’s 2D coordinates, θ denotes the heading angle of the robot, v denotes the linear speed, and $u _ { 1 } , u _ { 2 }$ denote the two control inputs for turning and acceleration. 

BarrierNet design. The robot is required to avoid a circular obstacle in its path, i.e, the state of the robot should satisfy: 

$$
(x - x _ {o}) ^ {2} + (y - y _ {o}) ^ {2} \geq R ^ {2}, \tag {19}
$$

where $( x _ { o } , y _ { o } ) \in \mathbb { R } ^ { 2 }$ denotes the location of the obstacle, and $R > 0$ is the radius of the obstacle. 

The goal is to minimize the control input effort, while subject to the safety constraint (19) as the robot approaches its destination, as shown in Fig. 6. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/1b4253a9-0808-41d3-b3e9-787ae7aa33f1/5b86117d860f314cba7723c3339d6b4777bfabbfa6c9231f4ee4b4135280f174.jpg)



Figure 6: A 2D robot navigation problem. The robot is required to avoid the obstacle in its path. The trajectories (the red and green ones) vary under different definitions of HOCBFs that enforce the safety cosntraint (19).


The relative degree of the safety constraint (19) is 2 with respect to the dynamics (18), thus, we use a HOCBF $b ( \pmb { x } ) = ( x - x _ { o } ) ^ { 2 } + ( y - y _ { o } ) ^ { 2 }$ − $R ^ { 2 }$ to enforce it. Any control input u should satisfy the HOCBF constraint (4) which in this case (choose $\alpha _ { 1 } , \alpha _ { 2 }$ in Def. 4 as linear functions) is: 

$$
- L _ {g} L _ {f} b (\boldsymbol {x}) \boldsymbol {u} \leq L _ {f} ^ {2} b (\boldsymbol {x}) + (p _ {1} (\boldsymbol {z}) + p _ {2} (\boldsymbol {z})) L _ {f} b (\boldsymbol {x}) + (\dot {p} _ {1} (\boldsymbol {z}) + p _ {1} (\boldsymbol {z}) p _ {2} (\boldsymbol {z})) b (\boldsymbol {x}) \tag {20}
$$

where 

$$
L _ {g} L _ {f} b (\pmb {x}) = [ - 2 (x - x _ {o}) v \sin \theta + 2 (y - y _ {o}) v \cos \theta , 2 (x - x _ {o}) \cos \theta + 2 (y - y _ {o}) \sin \theta ]
$$

$$
L _ {f} ^ {2} b (\boldsymbol {x}) = 2 v ^ {2} \tag {21}
$$

$$
L _ {f} b (\pmb {x}) = 2 (x - x _ {o}) v \cos \theta + 2 (y - y _ {o}) v \sin \theta
$$

In the above equations, $\pmb { z } = ( \pmb { x } , \pmb { x } _ { d } )$ is the input to the model, where $x _ { d } \in \mathbb { R } ^ { 2 }$ is the location of the destination, and $p _ { 1 } ( z ) , p _ { 2 } ( z )$ are the trainable penalty functions. ${ \dot { p } } _ { 1 } ( { \pmb x } )$ could be set as 0 due to the discretization solving method of the QP (Ames et al., 2017). 

The cost in the neuron of the BarrierNet is given by: 

$$
\min _ {\boldsymbol {u}} (u _ {1} - f _ {1} (\boldsymbol {z})) ^ {2} + (u _ {2} - f _ {2} (\boldsymbol {z})) ^ {2} \tag {22}
$$

where $f _ { 1 } ( z ) , f _ { 2 } ( z )$ are references controls provided by the upstream network (the outputs of the FC network). 

Results and dicussion. The training data is obtained by solving the CBF controller introduced in (Xiao and Belta, 2021), and we generate 100 trajectories of different destinations as the training data set. We compare the FC model, the deep foward-backward model (referred as DFB) (Pereira et al., 2020) that is equivalent to take the CBF-based QP as a safety filter, and our proposed BarrierNet. The training and testing results are shown in Figs. 7a-d. All the models are trained for obstacle size $R = 6 m$ . The controls from the BarrierNet can stay very close to the ground truth, while there are jumps for controls from the DFB when the robot gets close to the obstacle, which shows the conservativeness of the DFB, as shown by the blue solid (BarrierNet) and blue dashed (DFB) curves in Figs. 7a and 7b. The robot trajectory (dashed blue) from the DFB stays far away from ground truth in Fig. 7d, and this again shows its conservativeness. The robot from the FC will collide with the obstacle as there is no safety guarantee, as the dotted-blue line shown in Fig. 7d. 

When we increase the obstacle size during implementation (i.e., the trained BarrierNet/DFB/FC controller is used to drive a robot to its destination), the controls $u _ { 1 } , u _ { 2 }$ from the BarrierNet and DFB deviate from the ground true, as shown in Figs. 8a and 8b. This is due to the fact that the BarrierNet and DFB will always ensure safety first. Therefore, safety is always guaranteed in the BarrierNet and DFB, as the solid and dashed curves shown in Fig. 9a. Both the BarrierNet and DFB show some adaptivity to the size change of the obstacle. While the FC controller cannot be adaptive to the size change of the obstacle. Thus, the safety constraint (19) will be violated, as shown by the dotted curves in Fig. 9a. 

The difference between the DFB and the proposed BarrierNet is in the performance. In Fig. 9b, we show all the trajectories from the BarrierNet, DFB and FC controllers under different obstacle sizes. Collisions are avoided under the BarrierNet and DFB controllers, as shown by all the solid and dashed trajectories and the corresponding obstacle boundaries in Fig. 9b. However, as shown in Fig. 9b, the trajectories from the BarrierNet (solid) can stay closer to the ground true (red-solid) than the ones from the DFB (dashed) when $R = 6 m$ (and other R values). This is due to the fact that the CBFs in the DFB may not be properly defined such that the CBF constraint is active too early when the robot gets close to the obstacle. It is important to note that the robot does not have to stay close to the obstacle boundary under the BarrierNet controller, and this totally depends on the ground truth. The definitions of CBFs in the proposed BarrierNet depend on the environment (network input), and thus, they are adaptive, and are without conservativeness. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/1b4253a9-0808-41d3-b3e9-787ae7aa33f1/5b964ebeb70f020cd57c49a061c141c41951e96e603b82db7d78316c73a9a0f4.jpg)



(a) Control $u _ { 1 }$


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/1b4253a9-0808-41d3-b3e9-787ae7aa33f1/e3145af6cbfdcaf789bf30c12758a4dc8756a5a7d74cab4583a800788276dac1.jpg)



(b) Control $u _ { 2 } .$


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/1b4253a9-0808-41d3-b3e9-787ae7aa33f1/a0a77145ac9c100025bdbacae3f7405dd42512e0a893b19200c227ca666cfadd.jpg)



(c) Penalty functions.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/1b4253a9-0808-41d3-b3e9-787ae7aa33f1/5b794b058c32882ef6351ff7c71ba40aac8ee8edc0f20f15a4e90709e60a5f92.jpg)



(d) Robot trajectories.



Figure 7: The controls and trajectories from the FC, DFB and BarrierNet under the training obstacle size $R = 6 m$ . The results refer to the case that the trained FC/DFB/BarrierNet controller is used to drive a robot to its destination. Safety is guaranteed in both DFB and BarrierNet models, but not in the FC model. The DFB tends to be more conservative such that the trajectories/controls stay away from the ground true as its CBF parameters are not adaptive. The varying penalty functions allow the generation of desired control signals and trajectories (given by training labels), and demonstrate the adaptivity of the BarrierNet with safety guarantees.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/1b4253a9-0808-41d3-b3e9-787ae7aa33f1/03b31f01ee3fce3849bda086d186c34fc10bdd83434a39bfe6fceb8a965edae3.jpg)



(a) Control u1.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/1b4253a9-0808-41d3-b3e9-787ae7aa33f1/cf7cbbb83607e9300b7646c3381d2797271949242665256318cd5ef768bf6573.jpg)



(b) Control u2.



Figure 8: The controls from the BarrierNet and DFB under different obstacle sizes. The BarrierNet and DFB are trained under the obstacle size R = 6m. The results refer to the case that the trained BarrierNet/DFB controller is used to drive a robot to its destination. When we increase the obstacle size during implementation, the outputs (controls of the robot) of the BarrierNet and the DFB will adjust accordingly in order to guarantee safety, as shown by the blue and cyan curves. However, the BarrierNet tends to be less conservative for unseen situations.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/1b4253a9-0808-41d3-b3e9-787ae7aa33f1/a33a6d53393ac967905fd77aa8be3dfdcff9887a560d8f7a7c4368718dcee4ad.jpg)



(a) The HOCBF b(x) profiles under different (b) The robot trajectories under different obobstacle sizes. stacle sizes.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/1b4253a9-0808-41d3-b3e9-787ae7aa33f1/ce404eedf2a74ed2bb0b772d3513c3822ce77179d51ab6bb680f1009b9b71c63.jpg)



Figure 9: Safety metrics for the BarrierNet, the DFB and the FC network. The BarrierNet, the DFB and the FC network are trained under the obstacle size $R = 6 m . \ b ( { \pmb x } ) \geq 0$ implies safety guarantee. The trajectories under the FC controller coincide as the FC cannot adapt to the size change of the obstacle.


The profiles of the penalty functions $p _ { 1 } ( z ) , p _ { 2 } ( z )$ in the BarrierNet are shown in Fig. 7c. The values of the penalty functions vary when the robot approaches the obstacle and gets to its destination, and it shows the adaptivity of the BarrierNet in the sense that with the varying penalty functions, a BarrierNet can produce desired control signals given by labels (ground truth). This is due to the fact the varying penalty functions soften the HOCBF constraint without loosing safety guarantees. 

## 6.3. 3D Robot Navigation

Experiment setup. We consider a robot navigation problem with obstacle avoidance in 3D space. In this case, we consider complicated superquadratic safety constraints. The robot navigates according to the double integrator dynamics. The state of the robot is $\pmb { x } = ( p _ { x } , v _ { x } , p _ { y } , v _ { y } , p _ { z } , v _ { z } ) \in \mathbb { R } ^ { 6 }$ , in which the components denote the position and speed along $x , y , z { \mathrm { ~ a x e s } }$ , respectively. The three control inputs $u _ { 1 } , u _ { 2 } , u _ { 3 }$ are the acceleration along $x , y , z { \mathrm { ~ a x e s } }$ , respectively. 

BarrierNet design. The robot is required to avoid a superquadratic obstacle in its path, i.e, the state of the robot should satisfy: 

$$
\left(p _ {x} - x _ {o}\right) ^ {4} + \left(p _ {y} - y _ {o}\right) ^ {4} + \left(p _ {z} - z _ {o}\right) ^ {4} \geq R ^ {4}, \tag {23}
$$

where $( x _ { o } , y _ { o } , z _ { o } ) \in \mathbb { R } ^ { 3 }$ denotes the location of the obstacle, and $R > 0$ is the half-length of the superquadratic obstacle. 

The goal is to minimize the control input effort, while subject to the safety constraint (23) as the robot approaches its destination, as shown in Fig. 10. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/1b4253a9-0808-41d3-b3e9-787ae7aa33f1/4d6c13717711dc4d9bb2042adb63b965e9b91ce1aaa70a87b60875d5859410be.jpg)



Figure 10: A 3D robot navigation problem. The robot is required to avoid the obstacle in its path.


The relative degree of the safety constraint (23) is 2 with respect to the dynamics, thus, we use a HOCBF $b ( { \pmb x } ) = ( p _ { x } - x _ { o } ) ^ { 4 } + ( p _ { y } - y _ { o } ) ^ { 4 } + ( p _ { z } - z _ { o } ) ^ { 4 } - R ^ { 4 }$ to enforce it. Any control input u should satisfy the HOCBF constraint (4) which in this case (choose $\alpha _ { 1 } , \alpha _ { 2 }$ in Def. 4 as linear functions) is: 

$$
- L _ {g} L _ {f} b (\boldsymbol {x}) \boldsymbol {u} \leq L _ {f} ^ {2} b (\boldsymbol {x}) + (p _ {1} (\boldsymbol {z}) + p _ {2} (\boldsymbol {z})) L _ {f} b (\boldsymbol {x}) + (\dot {p} _ {1} (\boldsymbol {z}) + p _ {1} (\boldsymbol {z}) p _ {2} (\boldsymbol {z})) b (\boldsymbol {x}) \tag {24}
$$

where 

$$
L _ {g} L _ {f} b (\pmb {x}) = [ 4 (p _ {x} - x _ {o}) ^ {3}, \quad 4 (p _ {y} - y _ {o}) ^ {3}, \quad 4 (p _ {z} - z _ {o}) ^ {3} ]
$$

$$
L _ {f} ^ {2} b (\boldsymbol {x}) = 1 2 \left(p _ {x} - x _ {o}\right) ^ {2} v _ {x} ^ {2} + 1 2 \left(p _ {y} - y _ {o}\right) ^ {2} v _ {y} ^ {2} + 1 2 \left(p _ {z} - x _ {o}\right) ^ {2} v _ {z} ^ {2} \tag {25}
$$

$$
L _ {f} b (\pmb {x}) = 4 (p _ {x} - x _ {o}) ^ {3} v _ {x} + 4 (p _ {y} - y _ {o}) ^ {3} v _ {y} + 4 (p _ {z} - z _ {o}) ^ {3} v _ {z}
$$

In the above equations, $z = x$ is the input to the model, $p _ { 1 } ( z ) , p _ { 2 } ( z )$ are the trainable penalty functions. ${ \dot { p } } _ { 1 } ( { \pmb x } )$ is also set as 0 as in the 2D navigation case. 

The cost in the neuron of the BarrierNet is given by: 

$$
\min _ {\boldsymbol {u}} \left(u _ {1} - f _ {1} (\boldsymbol {z})\right) ^ {2} + \left(u _ {2} - f _ {2} (\boldsymbol {z})\right) ^ {2} + \left(u _ {3} - f _ {3} (\boldsymbol {z})\right) ^ {2} \tag {26}
$$

where $f _ { 1 } ( z ) , f _ { 2 } ( z ) , f _ { 3 } ( z )$ are references controls provided by the upstream network (the outputs of the FC network). 

Results and dicussion. The training data is obtained by solving the CBF controller introduced in (Xiao and Belta, 2021). We compare the FC model with our proposed BarrierNet. The training and testing results are shown in Figs. 11 and 12. The controls from the BarrierNet have some errors with repsect to the ground truth, and this is due to the complicated safety constraint (23). We can improve the tracking accuracy with deeper BarrierNet models (not the focus of this paper). Nevertheless, the implementation trajectory under the BarrierNet controller is close to the ground truth, as shown in Fig.12b. 

The robot is guaranteed to be collision-free from the obstacle under the BarrierNet controller, as the solid-blue line shown in Fig.12b. While the robot from the FC may collide with the obstacle as there is no safety guarantee, as the dotted-blue line shown in Fig. Fig.12b. The barrier function in Fig.12a also demonstrates the safety guarantees of the BarrierNet, but not in the FC model. The profiles of the penalty functions $p _ { 1 } ( z ) , p _ { 2 } ( z )$ in the BarrierNet are shown in Fig. 11d. The values of the penalty function variations demonstrate the adaptivity of the BarrierNet in the sense that with the varying penalty functions, a BarrierNet can produce desired control signals given by labels (ground truth). 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/1b4253a9-0808-41d3-b3e9-787ae7aa33f1/f4ffca55c021296aa6e54f035d6da0208fe37d2313dd289fcc207ae88b633004.jpg)



(a) Control $u _ { 1 }$


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/1b4253a9-0808-41d3-b3e9-787ae7aa33f1/da9f6f589422d6e64a0cc5f6947f6d02625e456f7de888ea182e6f34cacc0581.jpg)



(b) Control $u _ { 2 } .$


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/1b4253a9-0808-41d3-b3e9-787ae7aa33f1/62246fcf99884d65a2cd09964ded21ac67e753f6e6fc945a2a06f944b307cbbe.jpg)



(c) Control $u _ { 3 }$ .


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/1b4253a9-0808-41d3-b3e9-787ae7aa33f1/50fb41fc4c6b0565bcee89711646dbf15fe168714fc05d362f5a36c47cf24684.jpg)



(d) Penalty functions.



Figure 11: The controls and penalty functions from the and BarrierNet. The results refer to the case that the trained BarrierNet controller is used to drive a robot to its destination. The varying penalty functions allow the generation of desired control signals and trajectories (given by training labels), and demonstrate the adaptivity of the BarrierNet with safety guarantees.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/1b4253a9-0808-41d3-b3e9-787ae7aa33f1/f6d0fc8c99dcf779cceb2f1c20d20fea0d4f6e811eef4d0af1648fe73948b86b.jpg)



(a) The HOCBF b(x) profiles.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/1b4253a9-0808-41d3-b3e9-787ae7aa33f1/e5ef7fa62242b96ad108ea583c48e0d981de240affdd1ce971ffd4c19e06a348.jpg)



(b) The robot trajectories.



Figure 12: The HOCBFs and trajectories from the FC and BarrierNet. The results refer to the case that the trained FC/BarrierNet controller is used to drive a robot to its destination. Safety is guaranteed in the BarrierNet model, but not in the FC model.


## 7. Conclusion

In this work, we proposed BarrierNet - a differentiable HOCBF layer that is trainable and guarantees safety with respect to the user defined safe sets. BarrierNet can be integrated with any upstream neural network controller to provide a safety layer. In our experiments, we show that the proposed BarrierNet can guarantee safety while addressing the conservativeness that control barrier functions induce. A potential future avenue of research emerging from this work will be to simultaneously learn the system dynamics and unsafe sets with BarrierNets. This can be enabled using the expressive class of continuous-time neural network models (Chen et al., 2018; Lechner et al., 2020a; Hasani et al., 2021b; Vorbach et al., 2021). 

## Acknowledgments

This research was sponsored by the United States Air Force Research Laboratory and the United States Air Force Artificial Intelligence Accelerator and was accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the United States Air Force or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein. This work was further supported by The Boeing Company, and the Office of Naval Research (ONR) Grant N00014-18-1-2830. We are grateful to the members of the Capgemini research team for discussing the importance of safety and stability of machine learning. 

## References



Ramin Hasani, Mathias Lechner, Alexander Amini, Lucas Liebenwein, Max Tschaikowski, Gerald Teschl, and Daniela Rus. Closed-form continuous-depth models. arXiv preprint arXiv:2106.13898, 2021a. 





A. Robey, H. Hu, L. Lindemann, H. Zhang, D. V. Dimarogonas, S. Tu, and N. Matni. Learning control barrier functions from expert demonstrations. In 2020 59th IEEE Conference on Decision and Control (CDC), pages 3717–3724, 2020. 





S. Yaghoubi, G. Fainekos, and S. Sankaranarayanan. Training neural network controllers using control barrier functions in the presence of disturbances. In IEEE 23rd International Conference on Intelligent Transportation Systems (ITSC), pages 1–6, 2020. 





Aaron D Ames, Jessy W Grizzle, and Paulo Tabuada. Control barrier function based quadratic programs with application to adaptive cruise control. In 53rd IEEE Conference on Decision and Control, pages 6271–6278. IEEE, 2014. 





Ramin Hasani, Mathias Lechner, Alexander Amini, Daniela Rus, and Radu Grosu. Liquid timeconstant networks. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 7657–7666, 2021b. 





David E Rumelhart, Geoffrey E Hinton, and Ronald J Williams. Learning representations by backpropagating errors. nature, 323(6088):533–536, 1986. 





G. Yang, C. Belta, and R. Tron. Self-triggered control for safety critical systems using control barrier functions. In Proc. of the American Control Conference, pages 4454–4459, 2019. 





Aaron D Ames, Xiangru Xu, Jessy W Grizzle, and Paulo Tabuada. Control barrier function based quadratic programs for safety critical systems. IEEE Transactions on Automatic Control, 62(8): 3861–3876, 2017. 





Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016. 





M. Srinivasan, A. Dabholkar, S. Coogan, and P. A. Vela. Synthesis of control barrier functions using a supervised machine learning approach. In 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 7139–7145, 2020. 





H. Zhao, X. Zeng, T. Chen, and J. Woodcock. Learning safe neural network controllers with barrier certificates. Form Asp Comp, 33:437–455, 2021. 





B. Amos and J. Z. Kolter. Optnet: Differentiable optimization as a layer in neural networks. In Proceedings of the 34th International Conference on Machine Learning - Volume 70, pages 136– 145, 2017. 





Sepp Hochreiter and Jurgen Schmidhuber. Long short-term memory. ¨ Neural computation, 9(8): 1735–1780, 1997. 





Andrew Taylor, Andrew Singletary, Yisong Yue, and Aaron Ames. Learning for safety-critical control with control barrier functions. In Learning for Dynamics and Control, pages 708–717. PMLR, 2020a. 





Ricky TQ Chen, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud. Neural ordinary differential equations. In Proceedings of the 32nd International Conference on Neural Information Processing Systems, pages 6572–6583, 2018. 





W. Jin, Z. Wang, Z. Yang, and S. Mou. Neural certificates for safe control policies. preprint arXiv:2006.08465, 2020. 





Andrew J Taylor and Aaron D Ames. Adaptive safety with control barrier functions. In 2020 American Control Conference (ACC), pages 1399–1405. IEEE, 2020. 





Jason Choi, Fernando Castaneda, Claire J Tomlin, and Koushil Sreenath. Reinforcement learning ˜ for safety-critical control under model uncertainty, using control lyapunov functions and control barrier functions. In Robotics: Science and Systems (RSS), 2020. 





H. K. Khalil. Nonlinear Systems. Prentice Hall, third edition, 2002. 





Andrew J Taylor, Andrew Singletary, Yisong Yue, and Aaron D Ames. A control barrier perspective on episodic learning via projection-to-state safety. IEEE Control Systems Letters, 5(3):1019– 1024, 2020b. 





Noel Csomay-Shanklin, Ryan K Cosner, Min Dai, Andrew J Taylor, and Aaron D Ames. Episodic learning for safe bipedal locomotion with control barrier functions and projection-to-state safety. In Learning for Dynamics and Control, pages 1041–1053. PMLR, 2021. 





Mathias Lechner and Ramin Hasani. Learning long-term dependencies in irregularly-sampled time series. arXiv preprint arXiv:2006.04418, 2020. 





Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in neural information processing systems, pages 5998–6008, 2017. 





J. V. Deshmukh, J. P. Kapinski, T. Yamaguchi, and D. Prokhorov. Learning deep neural network controllers for dynamical systems with safety guarantees: Invited paper. In 2019 IEEE/ACM International Conference on Computer-Aided Design (ICCAD), pages 1–7, 2019. 





Mathias Lechner, Ramin Hasani, Alexander Amini, Thomas A Henzinger, Daniela Rus, and Radu Grosu. Neural circuit policies enabling auditable autonomy. Nature Machine Intelligence, 2(10): 642–652, 2020a. 





Charles Vorbach, Ramin Hasani, Alexander Amini, Mathias Lechner, and Daniela Rus. Causal navigation by continuous-time neural networks. arXiv preprint arXiv:2106.08314, 2021. 





J. Ferlez, M. Elnaggar, Y. Shoukry, and C. Fleming. Shieldnn: A provably safe nn filter for unsafe nn controllers. preprint arXiv:2006.09564, 2020. 





Mathias Lechner, Ramin Hasani, Daniela Rus, and Radu Grosu. Gershgorin loss stabilizes the recurrent neural network compartment of an end-to-end robot learning scheme. In 2020 IEEE International Conference on Robotics and Automation (ICRA), pages 5446–5452. IEEE, 2020b. 





Li Wang, Evangelos A Theodorou, and Magnus Egerstedt. Safe learning of quadrotor dynamics using barrier certificates. In 2018 IEEE International Conference on Robotics and Automation (ICRA), pages 2460–2465. IEEE, 2018. 





Sophie Gruenbacher, Jacek Cyranka, Mathias Lechner, Md Ariful Islam, Scott A Smolka, and Radu Grosu. Lagrangian reachtubes: The next generation. In 2020 59th IEEE Conference on Decision and Control (CDC), pages 1556–1563. IEEE, 2020. 





Mathias Lechner, Ramin Hasani, Radu Grosu, Daniela Rus, and Thomas A. Henzinger. Adversarial training is not ready for robot learning. In 2021 IEEE International Conference on Robotics and Automation (ICRA), pages 4140–4147, 2021. doi: 10.1109/ICRA48506.2021.9561036. 





W. Xiao and C. Belta. High order control barrier functions. In IEEE Transactions on Automatic Control, doi:10.1109/TAC.2021.3105491, 2021. 





Sophie Gruenbacher, Mathias Lechner, Ramin Hasani, Daniela Rus, Thomas A Henzinger, Scott Smolka, and Radu Grosu. Gotube: Scalable stochastic verification of continuous-depth models. arXiv preprint arXiv:2107.08467, 2021. 





Zhichao Li. Comparison between safety methods control barrier function vs. reachability analysis. arXiv preprint arXiv:2106.13176, 2021. 





W. Xiao and C. G. Cassandras. Decentralized optimal merging control for connected and automated vehicles with safety constraint guarantees. Automatica, 123:109333, 2021. 





Sophie Grunbacher, Ramin Hasani, Mathias Lechner, Jacek Cyranka, Scott A Smolka, and Radu Grosu. On the verification of neural odes with stochastic guarantees. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 11525–11535, 2021. 





B. T. Lopez, J. J. E. Slotine, and J. P. How. Robust adaptive control barrier functions: An adaptive and data-driven approach to safety. IEEE Control Systems Letters, 5(3):1031–1036, 2020. 





W. Xiao, C. Belta, and C. G. Cassandras. Event-triggered safety-critical control for systems with unknown dynamics. In Proc. of 60th IEEE Conference on Decision and Control, 2021a. 





Thomas Gurriet, Andrew Singletary, Jacob Reher, Laurent Ciarletta, Eric Feron, and Aaron Ames. Towards a framework for realizable safety critical control through active set invariance. In 2018 ACM/IEEE 9th International Conference on Cyber-Physical Systems (ICCPS), pages 98–106. IEEE, 2018. 





Pierre-Franc¸ois Massiani, Steve Heim, and Sebastian Trimpe. On exploration requirements for learning safety constraints. In Learning for Dynamics and Control, pages 905–916. PMLR, 2021. 





W. Xiao, C. Belta, and C. G. Cassandras. Adaptive control barrier functions. In IEEE Transactions on Automatic Control, DOI: 10.1109/TAC.2021.3074895, 2021b. 





Quan Nguyen and Koushil Sreenath. Exponential control barrier functions for enforcing high relative-degree safety-critical constraints. In 2016 American Control Conference (ACC), pages 322–328. IEEE, 2016. 





W. Xiao, C. G. Cassandras, and C. Belta. Bridging the gap between optimal trajectory planning and safety-critical control with applications to autonomous vehicles. Automatica, 129:109592, 2021c. 





M. A. Pereira, Z. Wang, I. Exarchos, and E. A. Theodorou. Safe optimal control using stochastic barrier functions and deep forward-backward sdes. In Conference on Robot Learning, 2020. 





Xiangru Xu, Paulo Tabuada, Jessy W Grizzle, and Aaron D Ames. Robustness of control barrier functions for safety critical control. IFAC-PapersOnLine, 48(27):54–61, 2015. 


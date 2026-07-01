# Differentiable Control Barrier Functions for Vision-based End-to-End Autonomous Driving

Wei Xiao∗, Tsun-Hsuan Wang∗, Makram Chahine, Alexander Amini, Ramin Hasani and Daniela Rus 

Abstract—Guaranteeing safety of perception-based learning systems is challenging due to the absence of ground-truth state information unlike in state-aware control scenarios. In this paper, we introduce a safety guaranteed learning framework for visionbased end-to-end autonomous driving. To this end, we design a learning system equipped with differentiable control barrier functions (dCBFs) that is trained end-to-end by gradient descent. Our models are composed of conventional neural network architectures and dCBFs. They are interpretable at scale, achieve great test performance under limited training data, and are safety guaranteed in a series of autonomous driving scenarios such as lane keeping and obstacle avoidance. We evaluated our framework in a sim-to-real environment, and tested on a real autonomous car, achieving safe lane following and obstacle avoidance via Augmented Reality (AR) and real parked vehicles. 

## I. INTRODUCTION

Neural networks are powerful tools for learning relevant representations in complex scenarios. However, applying such learning systems in decision and control problems such as autonomous driving is significantly hindered by the absence of safety assurance. This is due to the learning systems being a black box that makes it complex to perform root cause analysis. Thus, a single mistake made by a learned neural controller can potentially lead to catastrophic outcomes. Ensuring safety (e.g., providing guarantees that a self-driving car will never collide with obstacles) of a learning system is therefore very important. Nevertheless, as the dimensionality of the observations and action space for a real-world problem increases, defining safety criteria and guarantees for learning systems increases dramatically. Take for instance a visionbased autonomous driving algorithm; even identifying the contributions of every single pixel to making driving decisions in every scenario is computationally intractable. In this case, how can we ensure the safety of the learning system? 

In this paper we leverage theoretical results in differentiable control barrier functions (dCBF) to equip end-to-end visionbased learning systems with safety guarantees. Barrier functions (BFs) have been widely used in optimization formulations to guarantee the satisfaction of some constraints [26], and have recently been extended to Lyapunov-like functions [42, 47]. They have been employed to prove set invariance [7, 38, 48] and regulate multi-objective control [33]. Control Barrier Functions (CBFs) are extensions of BFs for control systems, and are used to map a constraint defined over system states onto a constraint on the control input [1]. The satisfaction of the control constraint thus implies the satisfaction of the original safety constraint. Recently, it has been shown that by optimizing a quadratic cost and satisfying state and control constraints, CBFs can be used to form quadratic programs (QPs) [32], [1], [45] which can be solved in real time. Prior work also introduced differentiable CBFs as novel CBF formulations whose parameters are trainable [51], [34]. They have been incorporated into differentiable QPs [5] which can in turn be combined with learning systems [53]. 


An End-to-End Learning System with a BarrierNet


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/907d4492-162c-41ca-b41d-d8e20ec4f506/4657c742815b5455c2a18f6567759e199cb24f9b0b4b40ef76613d1843a36ecf.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/907d4492-162c-41ca-b41d-d8e20ec4f506/4e6d72edb37ed4e40d521f8e73906b59cee6ebb184a8d05e0f7f0b1ddba5178a.jpg)



Fig. 1: Vision-based end-to-end autonomous driving with differentiable CBFs in a BarrierNet. Lane keeping and collision avoidance are guaranteed.


dCBFs are introduced to mitigate the conservativeness of CBFs for safety guarantees. In CBFs, a system tends to stay far away from the unsafe set boundary, and thus may deviate largely from a desired trajectory. A BarrierNet [53] is constructed by incorporating dCBFs into a differentiable QP. In a BarrierNet, all QP parameters, including those originating in CBFs, are trainable. Thus, we obtain a ubiquitous safety guaranteed barrier layer that can be combined with any learning system [24, 23, 44, 29, 21, 27, 20]. BarrierNet addresses the conservativeness of CBFs and allows the safety constraints of a neural controller to be adaptable to changing environments. 

Although BarrierNets show promise in guaranteeing safety for shallow neural controllers, they have not been studied in the context of high-dimensional observation spaces such as vision-based control. In this paper, we investigate the effectiveness and conditions for using BarrierNets in end-toend vision-based autonomous driving scenarios (see Fig. 1). In this example, a neural network receives a camera input stream and outputs acceleration and steering rate commands to navigate the vehicle along the center of the driving lane, while avoiding obstacles. The system and environment observations are inputs to the upstream network whose outputs serve as arguments to the BarrierNet layer. Finally, BarrierNet outputs the controls that guarantee collision avoidance. 

Contributions: (i) We present a new end-to-end learning pipeline composed of conventional deep learning models and BarrierNets for vision-based end-to-end autonomous driving to achieve safe maneuvering. (ii) Our algorithm is multilevel interpretable and can achieve good test performance under limited training data. (iii) We design a vision-based state estimation module within our pipeline and study how BarrierNet works in the absence of ground truth information (iv) We train and verify our proposed framework in a simto-real environment; we also deploy our model on a fullscale autonomous vehicle for safe lane following and obstacleavoidance via Augmented Reality (AR) and physically parked vehicles. 

## II. RELATED WORK

Safety-critical control and learning. A large body of work studied CBF-based safety guarantees [1, 32, 45, 12, 41]. Many existing works [1, 32, 57] combine CBFs for systems with quadratic costs to form optimization problems. Time is discretized and an optimization problem with constraints given by the CBFs is solved at each time step. Replacing CBFs by High Order Control Barrie Functions (HOCBFs) allows us to handle constraints with arbitrary relative degree [50]. The common observation is that CBFs tend to make the system’s behavior excessively conservative if they are not properly defined. This conservativeness is usually characterized by how close the system state can (but not necessarily) stay to the unsafe set boundary or a desired reference trajectory in obstacle-clustered environment. 

In order to address the conservativeness of the CBF method, the work in [51] proposed to parameterize the definition of CBFs, and use machine learning methods to learn CBF parameters for a certain type of unsafe sets such that safety is guaranteed without control being excessively conservative. This form of CBFs is differentiable. In [34] learning CBF parameters using a differential policy over a time horizon is proposed. Adaptive CBFs (AdaCBFs) [52] allow for timevarying CBF parameters. It has been shown that the satisfaction of the AdaCBF constraint is a necessary and sufficient condition for the satisfaction of the original safety constraint. All these related works require the design of proper policies and models, which is tedious and non-trivial. In contrast, we propose to use an end-to-end differentiable framework to learn neural network controllers for safety-critical systems. 

At the intersection of CBFs and learning, supervised learning techniques have been proposed to learn safe set definitions from demonstrations [39] which can be then enforced by CBFs. The authors in [41] used data to learn system dynamics for CBFs. In [56], neural network controllers are trained using CBFs in the presence of disturbances. These prior works focus on learning safe sets and dynamics, whereas we focus on the design of environment dependent and trainable CBFs. 

Differentiable optimization-based safety frameworks. Recent advances in differentiable optimization methods show promise for safety guaranteed neural network controllers [5, 36, 6, 53]. In [5], a differentiable quadratic program (QP) layer, called OptNet, was introduced. OptNet with CBFs has been used in neural networks as a filter for safe controls [36], but OptNet is not trainable, thus, potentially limiting the system’s learning performance. In [14, 25, 59, 16, 17], safety guaranteed neural network controllers have been learned through verification-in-the-loop training. A safe neural network filter has been proposed in [15] for a specific vehicle model using verification methods. The verification approaches cannot ensure coverage of the entire state space. They are offline methods, unable to adapt to environment changes (such as varying size of the unsafe sets) [31]. By comparison, the BarrierNet in [53] incorporates differentiable CBFs into neural network controllers. In this work, we address the challenges of using BarrierNet to achieve vision-based endto-end autonomous driving. 

Sim-to-real end-to-end autonomous driving. Many works have demonstrated the ability to learn end-to-end (perceptionto-control) driving policies directly from real-world perception data using imitation learning (IL) [37, 9]. Such approaches have been demonstrated to be successfully deployed in both real-world offline passive datasets [55, 43, 2] as well as real closed-loop control test environments [35, 13, 29, 22]. 

However, these works have focused on “reactive” scenarios such as lane-keeping [9, 55, 43], lane-changing [10], and navigation [35, 13, 2, 22], but lack the ability to plan a path around other objects in the scene due to significantly larger data requirements of IL to achieve sufficient coverage of the testing distribution. Simulation has emerged as a viable candidate to overcome this challenge and render a continuum of scenarios for learning in the presence of other objects and agents in the environment. Works that learn object avoidance in simulation have leveraged both imitation learning [54] as well as reinforcement learning [8, 28, 11, 58, 19] but often face limited to no deployment capabilities in reality due to large sim-to-real gaps present in model-based simulation. In this work, we leverage recent advances in data-driven simulation [3, 30, 46, 4] to overcome the sim-to-real gap to learn robust end-to-end controllers capable of transferring to real scenarios with other agents. 

## III. BACKGROUND

In this section, we briefly introduce control barrier functions (CBF) and refer interested readers to [1] for detailed formulations. Intuitively, CBFs are a means to translate state constraints to control constraints under affine dynamics. The controls that satisfy those constraints can be efficiently solved for by formulating a quadratic program. We start with the definition of class K functions. 

Definition 1: (Class K function [26]) A continuous function $\alpha : [ 0 , a ) \to [ 0 , \infty ) , a > 0$ is said to belong to class K if it is strictly increasing and $\alpha ( 0 ) = 0 .$ A continuous function $\beta : \mathbb { R } $ R is said to belong to extended class K if it is strictly increasing and $\beta ( 0 ) = 0$ . 

Consider an affine control system of the form 

$$
\dot {\boldsymbol {x}} = f (\boldsymbol {x}) + g (\boldsymbol {x}) \boldsymbol {u} \tag {1}
$$

where $\pmb { x } \in \mathbb { R } ^ { n } , \ f \ : \mathbb { R } ^ { n } \to \ \mathbb { R } ^ { n }$ and $g \ : \ \mathbb { R } ^ { n } \ \to \ \mathbb { R } ^ { n \times q }$ are locally Lipschitz, and $\pmb { u } \in U \subset \mathbb { R } ^ { q }$ , where U denotes a control constraint set. 

Definition 2: A set $C \subset \mathbb { R } ^ { n }$ is forward invariant for system (1) if its solutions for some $\mathbf { \pmb { { u } } } \in U$ starting at any $\pmb { x } ( 0 ) \in C$ satisfy ${ \pmb x } ( t ) \in C , \forall t \geq 0$ . 

Definition 3: (Relative degree) The relative degree of a (sufficiently many times) differentiable function $b : \mathbb { R } ^ { n } $ R with respect to system (1) is the number of times it needs to be differentiated along its dynamics until the control u explicitly shows in the corresponding derivative. 

Since function b is used to define a (safety) constraint $b ( { \pmb x } ) \geq$ 0, we will also refer to the relative degree of b as the relative degree of the constraint. For a constraint $b ( { \pmb x } ) \geq 0$ with relative degree m, $b : \mathbb { R } ^ { n }  \mathbb { R } .$ and $\psi _ { 0 } ( { \pmb x } ) : = b ( { \pmb x } )$ , we define a sequence of functions $\psi _ { i } : \mathbb { R } ^ { n }  \mathbb { R } , i \in \{ 1 , \dots , m \}$ : 

$$
\psi_ {i} (\boldsymbol {x}) := \dot {\psi} _ {i - 1} (\boldsymbol {x}) + \alpha_ {i} \left(\psi_ {i - 1} (\boldsymbol {x})\right), \quad i \in \{1, \dots , m \}, \tag {2}
$$

where $\alpha _ { i } ( \cdot ) , i \in \{ 1 , \ldots , m \}$ denotes a $( m \mathrm { ~ - ~ } i ) ^ { t h }$ order differentiable class K function. 

We further define a sequence of sets $C _ { i } , i \in \{ 1 , \ldots , m \}$ associated with (2) in the form: 

$$
C _ {i} := \{\boldsymbol {x} \in \mathbb {R} ^ {n}: \psi_ {i - 1} (\boldsymbol {x}) \geq 0 \}, \quad i \in \{1, \dots , m \}. \tag {3}
$$

Definition 4: (High Order Control Barrier Function (HOCBF) [50]) Let $C _ { 1 } , \ldots , C _ { m }$ be defined by (3) and $\psi _ { 1 } ( \pmb { x } ) , \ldots , \psi _ { m } ( \pmb { x } )$ be defined by (2). A function $b : \mathbb { R } ^ { n }  \mathbb { R }$ is a High Order Control Barrier Function (HOCBF) of relative degree m for system (1) if there exist $( m \mathrm { ~ - ~ } i ) ^ { t h }$ order differentiable class K functions $\alpha _ { i } , i \in \{ 1 , \ldots , m - 1 \}$ and a class K function $\alpha _ { m }$ such that 

$$
\begin{array}{l} \sup _ {\boldsymbol {u} \in U} \left[ L _ {f} ^ {m} b (\boldsymbol {x}) + \left[ L _ {g} L _ {f} ^ {m - 1} b (\boldsymbol {x}) \right] \boldsymbol {u} + O (b (\boldsymbol {x})) \right. \tag {4} \\ \left. \right.\left. + \alpha_ {m} \left(\psi_ {m - 1} (\boldsymbol {x})\right)\right] \geq 0, \\ \end{array}
$$

for all $\pmb { x } \in C _ { 1 } \cap , . . . , \cap C _ { m }$ . In (4), $L _ { f } ^ { m } ~ ( L _ { g } )$ denotes Lie derivatives along $\textit { f } \left( g \right)$ m (one) times, and $O ( b ( { \pmb x } ) ) =$ $\begin{array} { r } { \sum _ { i = 1 } ^ { m - 1 } L _ { f } ^ { i } ( \alpha _ { m - i } \circ \psi _ { m - i - 1 } ) ( { \pmb x } ) } \end{array}$ $L _ { g } L _ { f } ^ { m - 1 } \bar { b } ( { \pmb x } ) \neq 0$ on the boundary of the set $C _ { 1 } \cap , \ldots , \cap C _ { m } .$ 

Note that by setting $m = 1$ in a HOCBF, we can get a relative-degree-one CBF constraint: 

$$
L _ {f} b (\boldsymbol {x}) + L _ {g} b (\boldsymbol {x}) + \alpha_ {1} (b (\boldsymbol {x})) \geq 0. \tag {5}
$$

Theorem 1: ([50]) Given a HOCBF $b ( { \pmb x } )$ from Def. 4 with the associated sets $C _ { 1 } , \ldots , C _ { m }$ defined by (3), if $\pmb { x } ( 0 ) \in C _ { 1 } \cap , . . . , \cap C _ { m }$ , then any Lipschitz continuous controller $\boldsymbol {u} (t)$ that satisfies the constraint in (4), $\forall t \geq 0$ renders $C _ { 1 } \cap , \ldots , \cap C _ { m }$ forward invariant for system (1).

Combining CBFs with a quadratic cost $\begin{array} { r } { \int _ { t _ { 0 } } ^ { t _ { f } } \pmb { u } ^ { T } ( t ) H \pmb { u } ( t ) } \end{array}$ R tft uT (t)H u(t), 0 where H is positive definite, we can formulate CBF-based QPs: 

$$
\boldsymbol {u} ^ {*} (t) = \arg \min _ {\boldsymbol {u} (t)} \frac {1}{2} \boldsymbol {u} (t) ^ {T} H \boldsymbol {u} (t) \tag {6}
$$

s.t. 

$$
L _ {f} ^ {m} b (\boldsymbol {x}) + \left[ L _ {g} L _ {f} ^ {m - 1} b (\boldsymbol {x}) \right] \boldsymbol {u} + O (b (\boldsymbol {x})) + \alpha_ {m} \left(\psi_ {m - 1} (\boldsymbol {x})\right) \geq 0
$$

$$
\boldsymbol {u} \in U, \quad t = k \Delta t + t _ {0},
$$

## IV. PROBLEM FORMULATION

We now formally define learning of safety-critical control for autonomous driving. 

Problem 1: Given (i) front-view RGB camera images of the vehicle, (ii) a nominal controller $k ^ { \star } ( { \pmb x } ) = { \pmb u } ^ { \star }$ (such as a model predictive controller) (iii) vehicle dynamics in the form of (1), (iv) a set of safety constraints $b _ { j } ( \pmb { x } ) \geq 0 , j \in S$ (where $b _ { j }$ is continuously differentiable). Typical safety constraints include obstacle avoidance and lane keeping. (v) control bounds ${ \pmb u } _ { m i n } \le { \pmb u } \le { \pmb u } _ { m a x }$ of the vehicle, and (vi) a neural network controller $k ( \pmb { x } | \theta ) = \pmb { u }$ parameterized by θ, our goal is to take (i) as input and find the optimal parameters 

$$
\theta^ {\star} = \underset {\theta} {\arg \min} \mathbb {E} _ {\boldsymbol {x}} [ l (k ^ {\star} (\boldsymbol {x}), k (\boldsymbol {x} | \theta)) ] \tag {7}
$$

while guaranteeing the satisfaction of the safety constraints in (iv) and control bounds in (v). E(·) is the expectation and $l ( \cdot , \cdot )$ denotes a similarity measure. Note that state estimation required in (iii) and (iv) is implicitly done in the neural network by inferring from the vision inputs (i). 

## V. SAFETY-AWARE DIFFERENTIABLE FRAMEWORK

In this section, we propose a safety-guaranteed neural network controller for vision-based end-to-end autonomous driving based on BarrierNet. We first study where the two ”ends” should be defined in this framework in order to learn a good model based on limited data. 

## A. Interpretable End-to-End Design

System’s Inputs Setup. In human-driving, the majority of the information human drivers rely on comes from the front vision view. Therefore, we let the input of the end-to-end architecture be front view images, which contain enough information for executing safe driving. 

System’s Outputs Setup At the output end, where BarrierNet is implemented, high-relative-degree control variables (such as acceleration, jerk, steering rate or steering acceleration) are generated for driving the vehicle. The first advantage of using high-relative-degree control variables is to ensure the smoothness of the vehicle states (such as speed), which ensure the vehicle is maneuvered in a smooth manner so that passenger comfort is met. Another advantage is to ensure the controller works with accurate maneuvering due to the physical inertia of the vehicle. If we take vehicle speed as one of the controls, and the controller requires the speed to suddenly change to a large different value, the vehicle powertrain system will fail to respond. In this case the vehicle control becomes inaccurate, affecting the performance and even the safety of the vehicle. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/907d4492-162c-41ca-b41d-d8e20ec4f506/cf43a91ed8affc16d48ec6eab6196f5fb99b038a60382ff45856bb55e43012cf.jpg)



Fig. 2: Multi-level interpretable end-to-end autonomous driving framework with differentiable CBFs. The entire pipeline is end-to-end differentiable. Each depth of the model learns different vehicle and environment information. The outputs of BarrierNet are high-relative-degree controls.


The main challenge in an end-to-end autonomous driving with high-relative-degree control variables is that it requires a very large training data set with high diversity. This is because more vehicle state variables are involved in higher-relativedegree controls, and thus, the vehicle may have different states under the same observation. This will cause confusion in the training process. If the training dataset is not large and diverse enough, the poor generalization of the trained controller (from open-loop to closed-loop, or from sim-to-real) would make the controller fail to achieve its task. Thus, we propose a multilevel interpretable model under limited training data set as shown next. 

Setting up an interpretable framework. In order to make the model interpretable, we take training loss outputs at different depths of the model. Following the CNN or LSTM, we may take part of the neurons as the loss outputs for the locations of the vehicle itself and the obstacles. This way, we train neurons at this level to learn position information. In a deeper setting, we may take part of the neurons as another loss outputs for the speed and steering angle of the vehicle, training these neurons to learn speed and steering angle information. By adding derivative layers following these neurons, we get acceleration and steering rate information of the vehicle. The acceleration and steering rate could be taken as reference controls in the BarrierNet which is also trainable, and we take the output of the BarrierNet as the final loss output in the training process. In this framework, different depths of neurons learn different vehicle information, which makes the whole structure consistent with the vehicle physical dynamics. This architecture ensures the neural network model is interpretable [18]. We present the whole model structure in Fig. 2. 

## B. Differentiable Control Barrier Functions

In this subsection, we briefly introduce BarrierNet [53] that incorporates differentiable CBFs, and propose solutions for some of the challenges that arise in autonomous driving with BarrierNet. 

Differentiable CBFs are motivated by the fact that the traditional CBFs can easily make the system overly conservative. In order to address this conservativeness, we multiply the class K functions in (2) in the definition of a HOCBF with some observation-dependent functions $p _ { i } ( z ) , i \in \{ 1 , . . . , m \}$ : 

$$
\psi_ {i} (\boldsymbol {x}, \boldsymbol {z}) := \dot {\psi} _ {i - 1} (\boldsymbol {x}, \boldsymbol {z}) + p _ {i} (\boldsymbol {z}) \alpha_ {i} \left(\psi_ {i - 1} (\boldsymbol {x}, \boldsymbol {z})\right), \tag {8}
$$

$$
i \in \{1, \dots , m \},
$$

where $\psi _ { 0 } ( { \pmb x } , z ) = b ( { \pmb x } )$ and $z \in \mathbb { R } ^ { d }$ is the input (such as front view images in autonomous driving) of the neural network $( d \in \mathbb { N }$ is the dimension of the features), $p _ { i } : \mathbb { R } ^ { d }  \mathbb { R } ^ { > 0 } , i \in$ $\{ 1 , \ldots , m \}$ are the outputs of the previous layer, where $\mathbb { R } { > } 0$ denotes the set of positive scalars. The above formulation is similar to that of AdaCBF [52], but, contrary to the latter, is trainable and does not require the design of auxiliary dynamics for $p _ { i }$ (a non-trivial process). To ensure the validity of the above defined CBFs, we require that each $p _ { i }$ be continuously differentiable. Then, we have a differentiable HOCBF as in Def. 4 in the form: 

$$
L _ {f} ^ {m} b (\boldsymbol {x}) + \left[ L _ {g} L _ {f} ^ {m - 1} b (\boldsymbol {x}) \right] \boldsymbol {u} + O (b (\boldsymbol {x}), \boldsymbol {z}) \tag {9}
$$

$$
+ p _ {m} (\boldsymbol {z}) \alpha_ {m} \left(\psi_ {m - 1} (\boldsymbol {x}, \boldsymbol {z})\right) \geq 0,
$$

Note that it is possible to add additional training parameters to the above class K functions. For example, we may take the powers as training parameters if the class K functions are power functions. However, this may decrease the stability of the system as the values of the class K functions are more sensitive to powers than to coefficients. 

BarrierNet. Eventually, we can incorporate the above softened HOCBFs into differentiable QPs, and obtain a BarrierNet: 

$$
\boldsymbol {u} ^ {*} (t) = \arg \min _ {\boldsymbol {u} (t)} \frac {1}{2} \boldsymbol {u} (t) ^ {T} H (\boldsymbol {z} | \theta_ {h}) \boldsymbol {u} (t) + F ^ {T} (\boldsymbol {z} | \theta_ {f}) \boldsymbol {u} (t) \tag {10}
$$

s.t. 

$$
L _ {f} ^ {m} b _ {j} (\boldsymbol {x}) + \left[ L _ {g} L _ {f} ^ {m - 1} b _ {j} (\boldsymbol {x}) \right] \boldsymbol {u} + O \left(b _ {j} (\boldsymbol {x}), \boldsymbol {z} \mid \theta_ {p}\right) \tag {11}
$$

$$
+ p _ {m} (\boldsymbol {z} | \theta_ {p} ^ {m}) \alpha_ {m} (\psi_ {m - 1} (\boldsymbol {x}, \boldsymbol {z} | \theta_ {p})) \geq 0, j \in S
$$

$$
\boldsymbol {u} _ {\text { min }} \leq \boldsymbol {u} \leq \boldsymbol {u} _ {\text { max }},
$$

$$
t = k \Delta t + t _ {0},
$$

where $F ( z | \theta _ { f } ) \ \in \ \mathbb { R } ^ { q }$ could be interpreted as a reference control (can be the output of previous network layers) and $\theta _ { h } , \theta _ { f } , \theta _ { p } = ( \theta _ { p } ^ { 1 } , \ldots , \theta _ { p } ^ { m } )$ are trainable parameters. S denotes a set of safety constraints including obstacle avoidance and lane keeping. The above differentiable QPs formulate a neuron in BarrierNet. We let both $H ( z | \theta _ { h } )$ and $F ( z | \theta _ { f } )$ be parameterized and dependent on the network input $z ,$ but H and $F$ can also be directly trainable parameters that do not depend on the previous layer (i.e., we have H and $F )$ . The same applies to $p _ { i } , i \in \{ 1 , \ldots , m \}$ . The trainable parameters are $\theta = \{ \theta _ { h } , \theta _ { f } , \theta _ { p } \}$ (or $\theta = \{ H , F , p _ { i } , \forall i \in \{ 1 , \dots , m \} \}$ if $H , F$ and $p _ { i }$ do not depend on the previous layer). The solution $\boldsymbol { \mathbf { \mathit { u } } } ^ { * }$ is the output of the neuron. The BarrierNet layer is differentiable with respect to its parameters [5]. 

Safety with an unknown number of constraints. One of the challenges in autonomous driving with BarrierNet is that we have to define the exact number of the HOCBFs when designing the BarrierNet layer as it connects with previous layers. However, a vehicle may encounter time-varying number of obstacles (constraints) in a complex environment. In order to address this problem, we proceed as follows. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/907d4492-162c-41ca-b41d-d8e20ec4f506/08452ea95001ac1de2cc8be8c1b8705bca0b513aa983470d5bb098fc20b079d4.jpg)



Fig. 3: Large disk covering approach for obstacle avoidance. Collision can be avoided if the center of the vehicle never enters the disks (for a correct design of disk locations and sizes). Sorted disks are used to cover corresponding sorted obstacles as they present themselves.


Suppose $N \in  { \mathbb { N } }$ denotes the maximum number of obstacles (such as other vehicles) a vehicle may encounter in driving. We cover each of the obstacles with an off-the-center disk, as shown in Fig. 3. The deviation direction of the disk depends on the direction of the obstacle with respect to the lane center. In this manner, we may use large disks to cover obstacles while making sure that the ego vehicle will not be overly conservative in driving through. We may use multiple small disks to cover a single obstacle. However, this increases the number of safety constraints required. Another advantage of using a large off-the-center disks is to ensure the smoothness of the vehicle trajectory and to avoid getting stuck in local traps that may appear with small disks. In this setting, we only have N safety constraints, one for each obstacle. We sort them in a specific order in the connection with the previous layer, and enforce them using the above differentiable HOCBFs. 

When there are no actual obstacles on the road, as in the case depicted on the left-hand side of Fig. 3, we just move the covering disks off the road in which case they play the role of lane keeping. These disks move along the road as the vehicle progresses at the same speed. While the vehicle drives on the road, these disks do not affect its motion as the corresponding HOCBF constraints are not activated. However, if the vehicle is about to leave the road, these constraints can prevent it from doing so due to the safety guarantees of HOCBFs. 

When there is one or more obstacles on the road, as in the case depicted on the right-hand side of Fig. 3, we first sort the obstacles according to their distance with respect to the ego vehicle. Then, we use the sorted disks to cover the corresponding sorted obstacles. The sorted covering approach can make sure that vehicle may leave the road in order to avoid collision with obstacles. In this setting, although we may have redundant differentiable HOCBFs in terms of obstacle avoidance, these HOCBFs always play an important role in guiding the vehicle, either in lane keeping or obstacle avoidance. 

Tackling potential conflicts of HOCBF constraints and control bounds. Another challenge for BarrierNet is that the differentiable QPs can easily become infeasible to solve during training due to the possible conflict between the HOCBF constraints and the control bounds. In order to address this, we need to require that the nominal control provides control labels that strictly satisfy the safety constraints and control bounds. Then, during training in the BarrierNet layer, we can relax/remove control bounds. After the neural network converges, the differentiable QPs would be feasible when we add control bounds in the testing or implementation. However, there is still possibility that the QP could be infeasible as the BarrierNet may have some inputs that it has never seen before. In order to address this, we can find sufficient conditions of feasibility, as shown in [49]. Briefly, this approach finds a feasibility constraint on the state of system and the penalties $p _ { i } ( z ) , i \in \{ 1 , . . . , m \}$ , and then enforces this feasibility constraint using another CBF. 

## VI. EXPERIMENTS

In this section, we show experiments with the proposed vision-based end-to-end autonomous driving framework in both sim-to-real environments and a full-scale autonomous vehicle. We start by introducing the hardware platform and data collection, followed by implementation details of the proposed model. We then demonstrate extensive analysis in the sim-to-real environment VISTA [4]. Finally, we showcase results with real-car deployment. 

## A. Hardware Setup and Real-world Data Collection

We deploy our models onboard a full-scale autonomous vehicle (2019 Lexus RX 450H) equipped with a NVIDIA 2080Ti GPU and an AMD Ryzen 7 3800X 8-Core Processor. We use a RGB camera BFS-PGE-23S3C-CS as the primary perception sensor, which runs in ${ 3 0 } \mathrm { H z } ,$ with a resolution of 960×600, and has $1 3 0 ^ { \circ }$ horizontal field-of-view. Other onboard sensors include inertial measurement sensor (IMUs) and wheel encoders to measure steering feedback and odometry. Also, we use a differential global positioning system (dGPS) for evaluation purpose. To run the data-driven simulation VISTA [4], we collect real-world data from a wide-range of environments, including different time of day, weather conditions, and seasons of a year. The entire dataset consists of roughly 2 hour of driving data, which is further augmented with our training dataset generation pipeline using VISTA. 

## B. Synthetic Training Dataset Generation

We train our model with guided policy learning, which has been shown to improve effectiveness for direct model transfer to real-car deployment. The data generation process follows (a) in VISTA, randomly initializing both ego- and ado-car with different configurations like relative poses, geographical locations associated with the real dataset, appearance of the vehicle, etc (b) running an optimal controller with access to privileged information to steer the ego-vehicle and collect ground-truth control outputs with corresponding states, and (c) collecting RGB images at viewpoints along the trajectories. We choose nonlinear Model Predictive Control (NMPC) as the privileged (nominal) controller. While NMPC is usually computationally expensive and hard to solve, it is tractable offline and, with jerk $u _ { j e r k }$ and steering acceleration $u _ { s t e e r }$ as controls, provides smooth acceleration a and steering rate w, which is used as learning targets in BarrierNet. Vehicle dynamics of NMPC and BarrierNet (1) are defined with respect to a reference trajectory [40]. It measures the alongtrajectory distance $s \in \mathbb R$ and the lateral distance $d \in \mathbb { R }$ of the vehicle Center of Gravity (CoG) with respect to the closest point on the reference trajectory, 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/907d4492-162c-41ca-b41d-d8e20ec4f506/7f402573546dcd02175d799eb64f6db7a9e307370d0da8dbf1672915f86b0ef9.jpg)



Fig. 4: Coordinates of ego w.r.t a reference trajectory.


$$
\underbrace {\left[ \begin{array}{c} \dot {s} \\ \dot {d} \\ \dot {\mu} \\ \dot {v} \\ \dot {a} \\ \dot {\delta} \\ \dot {\omega} \end{array} \right]} _ {\dot {\boldsymbol {x}}} = \underbrace {\left[ \begin{array}{c} \frac {v \cos (\mu + \beta)}{1 - d \kappa} \\ v \sin (\mu + \beta) \\ \frac {v}{l _ {r}} \sin \beta - \kappa \frac {v \cos (\mu + \beta)}{1 - d \kappa} \\ a \\ 0 \\ \omega \\ 0 \end{array} \right]} _ {f (\boldsymbol {x})} + \underbrace {\left[ \begin{array}{c c} 0 & 0 \\ 0 & 0 \\ 0 & 0 \\ 0 & 0 \\ 1 & 0 \\ 0 & 0 \\ 0 & 1 \end{array} \right]} _ {g (\boldsymbol {x})} \underbrace {\left[ \begin{array}{c} u _ {j e r k} \\ u _ {s t e e r} \end{array} \right]} _ {\boldsymbol {u}}, \tag {12}
$$

where $\mu$ is the vehicle local heading error determined by the difference of the global vehicle heading $\theta \in \mathbb { R }$ and the tangent angle $\phi \in \mathbb { R }$ of the closest point on the reference trajectory $( \mathrm { i . e . , } \theta = \phi + \mu )$ as shown in $\mathrm { F i g . } 4 ; v ,$ a denote the vehicle linear speed and acceleration; $\delta ,$ ω denote the steering angle and steering rate, respectively; κ is the curvature of the reference trajectory at the closest point; $l _ { r }$ is the length of the vehicle from the tail to the CoG; and $u _ { j e r k } , u _ { s t e e r }$ denote the two control inputs for jerk and steering acceleration (in the nominal controller). $\begin{array} { r } { \beta = \arctan \left( \frac { l _ { r } } { l _ { r } + l _ { f } } \tan \delta \right) } \end{array}$ , where $l _ { f }$ is the length of the vehicle from the head to the CoG. We set the receding horizon of the NMPC to 20 time steps during data sampling, and it is implemented in a virtual simulation environment in MATLAB. We augment the real-world dataset using VISTA and NMPC with synthetic obstacle avoidance and lane following data. In total, the training dataset has around 400k images. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/907d4492-162c-41ca-b41d-d8e20ec4f506/1453ff8e20e3fa9c5cc48c64ac6179dc2b3a6f1689f715ccc034bb16c495bda9.jpg)



Fig. 5: Lane following probabilistic comparisons of deviation from the lane center in a BarrierNet with/without lane keeping CBFs.


Implementation Details. Based on the aforementioned general framework in Sec. V-B, we specifically define dCBFs for lane following as bleflf $b _ { l f } ^ { l e f t } = d _ { l f } - d$ $b _ { l f } ^ { r i g h \acute { t } } = d _ { l f } + d$ (besides the disk covering lane following approach shown in the left case of Fig. 3 as we only study single obstacle avoidance), $\dot { b _ { o b s } } = \bar { \Delta } s ^ { 2 } + ( d - d _ { o b s } ) ^ { 2 } - r _ { D } ^ { 2 } .$ where $d _ { l f }$ is a preset bound, ∆s is relative progress between ego-car and obstacle, $r _ { D }$ is disk size, and $d _ { o b s }$ is the lateral displacement from lane center of the obstacle. We compute Lie derivative to construct dCBF constraints in QP with vehicle dynamics mentioned above. Overall, the model takes in a frontview image, infers speed v and steering angle δ to compute reference control with derivative and state $( d , d _ { o b s } , \mu , \Delta s )$ , predicts dCBFs parameters, and obtains final control $( a , \omega )$ by solving QP with dCBFs. The learning supervision includes Mean Squared Error loss on $v , \delta , d , d _ { o b s } , \mu , \Delta s , a , \omega .$ . We bound the derivative of v, δ (reference control) to stabilize learning. We cap loss on $\Delta s , d _ { o b s }$ when the obstacle is absent or too far away, to ensure states can be reasonably predicted. 

## C. Evaluation In Sim-to-Real Environments

Open-loop control error (i.e., difference between predicted and ground-truth control) has been shown to be a poor indicator to evaluate the performance of a driving policy since it only measures error around ground-truth trajectories and ignores accumulated errors that gradually drift the vehicle to out-of-distribution regions. Hereby, we presents closed-loop testing results in the sim-to-real environment VISTA [4]. 

Lane Keeping As Safety Constraints. In Fig. 5, we show the probability of ego-vehicle deviating away from the lane center for larger than 1m. We run 1000 episodes with maximal 200 steps if not crashed (off-lane more than 2m) prematurely. In each episode, the vehicle is randomly initialized at a point in the trace and we compute average deviation at every point to ensure sufficiently large sample size for the statistics. The model with lane keeping CBFs achieves significantly better performance since they can encourage the autonomous vehicle to stay close to the lane center by decreasing the boundary values due to the Lyapunov property of CBFs. 

Obstacle Avoidance. In Table I, we show crash rate and minimal clearance of models with or without BarrierNet and with or without access to ground-truth states. Minimal clearance is computed as the closest distance between polygons of ego- and ado-car within an episode. The introduction of obstacle avoidance dCBFs significantly reduces crash rate and increases clearance. The remaining failures mainly come from the imprecise or even erroneous state and obstacle information inferred from the front-view camera only. With access to ground-truth information (an ideal state estimator), the crash rate is close to yet not zero. This might be due to misaligned dynamics and inter-sampling effects of CBFs which have been extensively studied in CBFs [41, 52]. To look deeper into how BarrierNet improves safety distance, we plot the distribution of clearance larger than a varying threshold among all time steps in Fig. 6, where larger area under the curve indicates better safety. 


TABLE I: Crash rate and clearance with/without BarrierNet, using or not using ground truth obstacle information.


<table><tr><td>Method</td><td>Crash Rate ↓</td><td>Min. Clearance (m) ↑</td></tr><tr><td>w/o dCBF</td><td>0.53</td><td>0.43</td></tr><tr><td>w/ dCBF</td><td>0.28</td><td>0.55</td></tr><tr><td>w/ dCBF (with gt)</td><td>0.03</td><td>0.61</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/907d4492-162c-41ca-b41d-d8e20ec4f506/23e3895b98e7310f708760cd6e34390bf18fbd3199018693df5336473b9748ed.jpg)



Fig. 6: Line plot for clearance in obstacle avoidance with/without BarrierNet.


Furthermore, in Fig. 7, we investigate how imperfect state estimation introduces error in dCBFs and affects crash rate. We show the four output prediction from the state estimation module, including deviation from lane center of ego-car $d ,$ local heading error $\mu$ with respect to road curvature, relative progress along the road between ego-car and the obstacle $\Delta s$ , and lateral displacement of obstacle from lane center $d _ { o b s }$ . As expected, the overall trend shows increasing failure with larger state estimation error. The performance drops drastically with large $d _ { o b s }$ since roughly it indicates whether the obstacle is at the left or right with respect to the ego-car and thus has great influence within obstacle avoidance dCBFs. Note that bins at large error can have fewer samples (e.g., the tail in histogram of $d )$ and may lead to high variance in the estimator. We still keep the results for completeness. The results highlight the importance of handling state estimation error and suggest future research on uncertainty calibration, which will be the focus of our continued effort. 

BarrierNet Provides Safe Maneuvers. In Fig. 8, we benchmarked for different learning systems, with and without BarrierNet, to provide driving trajectories under the same configuration except for arbitrary initial pose on a road. We show 10 variants of initial states for each model. It can be observed that trajectories from the two models mostly align with each other in the beginning as the ego-vehicle starts with different lateral displacement from the lane center and tries to recover. Then, the two set of trajectories diverges while approaching the obstacle. This is the consequence of correction from the activated dCBFs over the reference unsafe control. With BarrierNet, the safety is guaranteed. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/907d4492-162c-41ca-b41d-d8e20ec4f506/eb877eae5d11f9e7d5da863c1a99239c42db37db0d9521c3679b1516a640ef2c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/907d4492-162c-41ca-b41d-d8e20ec4f506/964aa46cc0d864e27641be29514f491881b382d0ad3937a60b2800a7b8464633.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/907d4492-162c-41ca-b41d-d8e20ec4f506/1d7db31fa53693e769e54ec3592e87b0bbd0faa2d2f91602b8043c011f49e858.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/907d4492-162c-41ca-b41d-d8e20ec4f506/0a564f4018c77dd1cad2b061ad8ac5ad55cb27577e007ae8078a51f641914063.jpg)



Fig. 7: Statistics for crash rate in obstacle avoidance under different levels of prediction error from front vision view.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/907d4492-162c-41ca-b41d-d8e20ec4f506/7d1794c9c1b3ea1914fa01b3f31d9bf907c858625b3e7827ce1c213cf6338943.jpg)



Fig. 8: Vehicle trajectories in obstacle avoidance with/without BarrierNet in VISTA.


BarrierNet With Different Profiles. We also notice that the BarrierNet may learn different CBF parameters when the ego vehicle approaches an obstacle. In Fig. 9, we present two possible variations of penalty functions $p _ { 1 } ( z ) , p _ { 2 } ( z )$ when the ego vehicle is around an obstacle. The penalty functions $p _ { 1 } ( z ) , p _ { 2 } ( z )$ adapt to the obstacle when the ego vehicle is close to an obstacle, and they recover to some values when the ego leaves the obstacle. This shows the flexibility of the BarrierNet. Another observation is that the outputs of the BarrierNet tend to deviate from the reference controls (from the previous LSTM layer) when the ego vehicle is close to the obstacle. This shows the safety guarantee property of CBFs. In order to avoid this deviation, we need to improve the learned model with better reference controls and CBF parameters. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/907d4492-162c-41ca-b41d-d8e20ec4f506/aec0d216ad2d323dc355cf9d5ea457409cf5f12234b79125d2c9b1ef9c694606.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/907d4492-162c-41ca-b41d-d8e20ec4f506/8587dd2c9118b08ef93f5491ab5f6ed57b1ae20d48d7905cfc38dea7ac94c405.jpg)



Fig. 9: Penalty $p _ { 1 } ( z ) , p _ { 2 } ( z )$ variation in a dCBF (BarrierNet) when approaching an obstacle under two (different) trained BarrierNets. The relative degree of the safety constraint is two, and thus we have two CBF parameters in one CBF. The segments inside the dotted boxes denote intervals when the ego vehicle is near the obstacle. The box sizes are different as the ego has different speeds when passing the obstacle.


## D. Physical Autonomous Car Experiments

To verify the effectiveness of the proposed vision-based endto-end framework with dCBFs, we deploy the trained models on a full-scale autonomous driving car. The experiments are conducted in a test site with rural road type. We majorly test the algorithm with augmented reality (AR) and only perform minimal experiment with real-car obstacle for safety reasons. We use a pre-collected map of the test site and vehicle pose from the differential GPS (dGPS) to place virtual obstacles in the front of the ego-vehicle on road with AR. Note that the tested models are still using vision inputs only to steer the autonomous vehicle without any access to ground-truth state. Fig. 10 is an illustration of the real-car experimental setup. Another thing worth mentioning is that the scene is covered with snow at the time we conducted real-car experiment. The icy road surface at the track and heavy snow at the side of the road introduce tire slippage and pose additional challenges to our self-driving system. Also, the reflection of sunlight on the ice makes it hard to recognize road boundaries even from human judgement. With high-precision dGPS in the site (covariance < 1cm), we provides qualitative analysis with side-by-side comparison between models with and without BarrierNet. 

BarrierNet In Challenging Sharp Turns. In Fig. 11, we demonstrate driving trajectories of BarrierNet with and without lane keeping CBFs in sharp left and right turns. We show the footprint of vehicle through time and indicate forward direction with arrows. Without lane keeping CBFs (red), the car is more prone to get off-road, while roughly correct estimates of deviation from lane center (d) imposes an additional layer of safety with lane keeping CBFs (blue). 

Obstacle Avoidance In Real World. We also did experiments on the autonomous car in obstacle avoidance, as shown in Fig. 12. The first example (left) demonstrates that with reasonable reference control (both models successfully avoid the obstacle), the model with obstacle avoidance dCBFs (blue) creates more clearance to achieve better safety. The second example (right) highlights the effectiveness of BarrierNet (blue) when the reference control (red) fails to avoid the front car and requires correction from activated dCBF constraints. 

## VII. CONCLUSION

We proposed an end-to-end learning framework for obtaining safety-guaranteed perception-based autonomous driving agents. Our method is constructed by combining a deep neural model with a differentiable higher-order control barrier function to ensure safe lane-keeping and obstacle avoidance. We showed how various modules of our pipeline can contribute to the understanding of the learning system’s behavior while driving. Our sim-to-real and real-world experiments demonstrated the effectiveness of our approach in many driving scenarios where we could strictly reduce the probability of crash and interventions, when our pipeline is activated. 

We hope that our method can inspire future research on endowing real-world robot learning schemes with fundamental control theory modules to enhance interpretability, robustness and safety. 

## ACKNOWLEDGMENTS

This work was partially supported by Capgemini Engineering. This research was also sponsored by the United States Air Force Research Laboratory and the United States Air Force Artificial Intelligence Accelerator and was accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the United States Air Force or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein. 

## REFERENCES



[1] Aaron D Ames, Jessy W Grizzle, and Paulo Tabuada. Control barrier function based quadratic programs with application to adaptive cruise control. In 53rd IEEE Conference on Decision and Control, pages 6271–6278. IEEE, 2014. 





[2] Alexander Amini, Guy Rosman, Sertac Karaman, and Daniela Rus. Variational end-to-end navigation and localization. In 2019 International Conference on Robotics and Automation (ICRA), pages 8958–8964. IEEE, 2019. 





[3] Alexander Amini, Igor Gilitschenski, Jacob Phillips, Julia Moseyko, Rohan Banerjee, Sertac Karaman, and Daniela Rus. Learning robust control policies for end-to-end autonomous driving from data-driven simulation. IEEE Robotics and Automation Letters, 5(2):1143–1150, 2020. 





[4] Alexander Amini, Tsun-Hsuan Wang, Igor Gilitschenski, Wilko Schwarting, Zhijian Liu, Song Han, Sertac Karaman, and Daniela Rus. Vista 2.0: An open, data-driven simulator for multimodal sensing and policy learning for autonomous vehicles. arXiv preprint arXiv:2111.12083, 2021. 





[5] Brandon Amos and J. Zico Kolter. Optnet: Differentiable optimization as a layer in neural networks. In Proceedings of the 34th International Conference on Machine Learning - Volume 70, pages 136–145, 2017. 



![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/907d4492-162c-41ca-b41d-d8e20ec4f506/0306138bd44c91f316f077b35a1b58b7f20b0fd2bf132b8bcf0430160b8a9706.jpg)




circuits. In Proceedings of the 2020 International Conference on Machine Learning (ICML). JMLR. org, 2020. 





[47] Peter Wieland and Frank Allgower. Constructive safety ¨ using control barrier functions. In Proc. of 7th IFAC Symposium on Nonlinear Control System, 2007. 





[20] Ramin Hasani, Mathias Lechner, Alexander Amini, Lucas Liebenwein, Max Tschaikowski, Gerald Teschl, and Daniela Rus. Closed-form continuous-depth models. arXiv preprint arXiv:2106.13898, 2021. 





[48] Rafael Wisniewski and Christoffer Sloth. Converse barrier certificate theorem. In Proc. of 52nd IEEE Conference on Decision and Control, pages 4713–4718, Florence, Italy, 2013. 





[21] Ramin Hasani, Mathias Lechner, Alexander Amini, Daniela Rus, and Radu Grosu. Liquid time-constant networks. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 7657–7666, 2021. 





[49] W. Xiao, C. Belta, and C. G. Cassandras. Sufficient conditions for feasibility of optimal control problems using control barrier functions. Automatica, 135:109960, 2022. 




Fig. 10: An illustration of real-car experiments.




[22] Jeffrey Hawke, Richard Shen, Corina Gurau, Siddharth Sharma, Daniele Reda, Nikolay Nikolov, Przemysław Mazur, Sean Micklethwaite, Nicolas Griffiths, Amar Shah, and Alex Kendall. Urban Driving with Conditional Imitation Learning. In IEEE International Conference on Robotics and Automation (ICRA), 2020. 





[50] Wei Xiao and Calin Belta. Control barrier functions for systems with high relative degree. In Proc. of 58th IEEE Conference on Decision and Control, pages 474–479, Nice, France, 2019. 



![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/907d4492-162c-41ca-b41d-d8e20ec4f506/26ef817bee05a8db8f73388a1c636d0fe35553999f93ef0d1c4603213b5a3efd.jpg)




[23] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016. 





[51] Wei Xiao, Calin Belta, and Christos G. Cassandras. Feasibility guided learning for constrained optimal control problems. In Proc. of 59th IEEE Conference on Decision and Control, pages 1896–1901, 2020. 



![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/907d4492-162c-41ca-b41d-d8e20ec4f506/914b863dca808246a952d59b08ef50043476f7d80e7a2f865f1768ca513b7550.jpg)




[24] Sepp Hochreiter and Jurgen Schmidhuber. Long short- ¨ term memory. Neural computation, 9(8):1735–1780, 1997. 





[52] Wei Xiao, Calin Belta, and Christos G. Cassandras. Adaptive control barrier functions. In IEEE Transactions on Automatic Control, DOI: 10.1109/TAC.2021.3074895, 2021. 




Fig. 11: Two cases of experimental vehicle trajectories in obstacle avoidance with/without lane keeping CBFs in the BarrierNet. Tire slipping happens on the icy road.




[25] Wanxin Jin, Zhaoran Wang, Zhuoran Yang, and Shaoshuai Mou. Neural certificates for safe control policies. preprint arXiv:2006.08465, 2020. 





[53] Wei Xiao, Ramin Hasani, Xiao Li, and Daniela Rus. Barriernet: A safety-guaranteed layer for neural networks. preprint arXiv:2111.11277, 2021. 



![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/907d4492-162c-41ca-b41d-d8e20ec4f506/e2e078ea6432b54bcd6dc2f336bc982ae8c07814bca5ef558f332debd90df4ca.jpg)




[26] Hassan K. Khalil. Nonlinear Systems. Prentice Hall, third edition, 2002. 





[54] Yi Xiao, Felipe Codevilla, Akhil Gurram, Onay Urfalioglu, and Antonio M Lopez. Multimodal End-to-End ´ Autonomous Driving. arXiv:1906.03199, 2019. 



![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/907d4492-162c-41ca-b41d-d8e20ec4f506/5d8a6f576c9cfcc82675bac06564ff9c859acd4b396642da49f590150f52648a.jpg)




[27] Mathias Lechner and Ramin Hasani. Mixed-memory rnns for learning long-term dependencies in irregularly sampled time series. 2021. 





[55] Huazhe Xu, Yang Gao, Fisher Yu, and Trevor Darrell. End-to-End Learning of Driving Models from Large-Scale Video Datasets. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017. 




Fig. 12: Two cases of experimental vehicle trajectories in obstacle avoidance with/without BarrierNet. In the left case, the heavy snow by the road is preventing the vehicle from getting back to the road due to tire slipping, and thus the vehicle recovers slowly even when the steering wheel is at its left limit.




[28] Mathias Lechner, Ramin Hasani, Manuel Zimmer, Thomas A Henzinger, and Radu Grosu. Designing worm-inspired neural networks for interpretable robotic control. In 2019 International Conference on Robotics and Automation (ICRA), pages 87–94. IEEE, 2019. 





[56] Shakiba Yaghoubi, Georgios Fainekos, and Sriram Sankaranarayanan. Training neural network controllers using control barrier functions in the presence of disturbances. In IEEE 23rd International Conference on Intelligent Transportation Systems (ITSC), pages 1–6, 2020. 





[29] Mathias Lechner, Ramin Hasani, Alexander Amini, Thomas A Henzinger, Daniela Rus, and Radu Grosu. Neural circuit policies enabling auditable autonomy. Nature Machine Intelligence, 2(10):642–652, 2020. 





[57] Guang Yang, Calin Belta, and Roberto Tron. Selftriggered control for safety critical systems using control barrier functions. In Proc. of the American Control Conference, pages 4454–4459, 2019. 





[6] Brandon Amos, Ivan Dario Jimenez Rodriguez, Jacob Sacks, Byron Boots, and J. Zico Kolter. Differentiable mpc for end-to-end planning and control. In Proceedings of the 32nd International Conference on Neural Information Processing Systems, page 8299–8310. Curran Associates Inc., 2018. 





[30] Wei Li, Chengwei Pan, Rong Zhang, Jiaping Ren, Yuexin Ma, Jin Fang, Feilong Yan, Qichuan Geng, Xinyu Huang, Huajun Gong, et al. Aads: Augmented autonomous driving simulation using data-driven algorithms. arXiv:1901.07849, 2019. 





[58] Catherine Zeng, Jordan Docter, Alexander Amini, Igor Gilitschenski, Ramin Hasani, and Daniela Rus. Dreaming with transformers. 2022. 





[7] Jean-Pierre Aubin. Viability theory. Springer, 2009. 





[31] Zhichao Li. Comparison between safety methods control barrier function vs. reachability analysis. arXiv preprint arXiv:2106.13176, 2021. 





[59] Hengjun Zhao, Xia Zeng, Taolue Chen, Zhiming Liu, and Jim Woodcock. Learning safe neural network controllers with barrier certificates. Form Asp Comp, 33:437–455, 2021. 





[8] Steven Bohez, Tim Verbelen, Elias De Coninck, Bert Vankeirsbilck, Pieter Simoens, and Bart Dhoedt. Sensor Fusion for Robot Control through Deep Reinforcement Learning. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2017. 





[32] Quan Nguyen and Koushil Sreenath. Exponential control barrier functions for enforcing high relative-degree safety-critical constraints. In 2016 American Control Conference (ACC), pages 322–328. IEEE, 2016. 





[9] Mariusz Bojarski, Davide Del Testa, Daniel Dworakowski, Bernhard Firner, Beat Flepp, Prasoon Goyal, Lawrence D Jackel, Mathew Monfort, Urs Muller, Jiakai Zhang, et al. End to end learning for self-driving cars. arXiv preprint arXiv:1604.07316, 





[33] Dimitra Panagou, Dusan M. Stipanovi ˇ c, and Petros G. ˇ Voulgaris. Multi-objective control for multi-agent systems using lyapunov-like barrier functions. In Proc. of 52nd IEEE Conference on Decision and Control, pages 1478–1483, Florence, Italy, 2013. 





2016\. 





[34] Hardik Parwana and Dimitra Panagou. Recursive feasibility guided optimal parameter adaptation of differential convex optimization policies for safety-critical systems. preprint arXiv:2109.10949, 2021. 





[10] Mariusz Bojarski, Chenyi Chen, Joyjit Daw, Alperen Degirmenci, Joya Deri, Bernhard Firner, Beat Flepp, ˘ Sachin Gogri, Jesse Hong, Lawrence Jackel, et al. The nvidia pilotnet experiments. arXiv preprint arXiv:2010.08776, 2020. 





[35] Naman Patel, Anna Choromanska, Prashanth Krishnamurthy, and Farshad Khorrami. Sensor Modality Fusion with CNNs for UGV Autonomous Driving in Indoor Environments. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2017. 





[11] Axel Brunnbauer, Luigi Berducci, Andreas Brandstatter, ¨ Mathias Lechner, Ramin Hasani, Daniela Rus, and Radu Grosu. Latent imagination facilitates zero-shot transfer in autonomous racing. arXiv preprint arXiv:2103.04909, 2021. 





[36] Marcus Aloysius Pereira, Ziyi Wang, Ioannis Exarchos, and Evangelos A. Theodorou. Safe optimal control using stochastic barrier functions and deep forward-backward sdes. In Conference on Robot Learning, 2020. 





[12] Jason Choi, Fernando Castaneda, Claire J Tomlin, and˜ Koushil Sreenath. Reinforcement learning for safetycritical control under model uncertainty, using control lyapunov functions and control barrier functions. In Robotics: Science and Systems (RSS), 2020. 





[37] Dean A Pomerleau. Alvinn: An autonomous land vehicle in a neural network. Technical report, CARNEGIE-MELLON UNIV PITTSBURGH PA ARTIFICIAL IN-TELLIGENCE AND PSYCHOLOGY . . . , 1989. 





[13] Felipe Codevilla, Matthias Miiller, Antonio Lopez, ´ Vladlen Koltun, and Alexey Dosovitskiy. End-to-End Driving via Conditional Imitation Learning. In IEEE International Conference on Robotics and Automation (ICRA), 2018. 





[38] Stephen Prajna, Ali Jadbabaie, and George J. Pappas. A framework for worst-case and stochastic safety verification using barrier certificates. IEEE Transactions on Automatic Control, 52(8):1415–1428, 2007. 





[14] Jyotirmoy V. Deshmukh, James P. Kapinski, Tomoya Yamaguchi, and Danil Prokhorov. Learning deep neural network controllers for dynamical systems with safety guarantees: Invited paper. In 2019 IEEE/ACM International Conference on Computer-Aided Design (ICCAD), pages 1–7, 2019. 





[39] Alexander Robey, Haimin Hu, Lars Lindemann, Hanwen Zhang, Dimos V. Dimarogonas, Stephen Tu, and Nikolai Matni. Learning control barrier functions from expert demonstrations. In 2020 59th IEEE Conference on Decision and Control (CDC), pages 3717–3724, 2020. 





[15] James Ferlez, Mahmoud Elnaggar, Yasser Shoukry, and Cody Fleming. Shieldnn: A provably safe nn filter for unsafe nn controllers. preprint arXiv:2006.09564, 2020. 





[40] Alessandro Rucco, Giuseppe Notarstefano, and John Hauser. An efficient minimum-time trajectory generation strategy for two-track car vehicles. IEEE Transactions on Control Systems Technology, 23(4):1505–1519, 2015. 





[16] Sophie Gruenbacher, Ramin Hasani, Mathias Lechner, Jacek Cyranka, Scott A Smolka, and Radu Grosu. On the verification of neural odes with stochastic guarantees. arXiv preprint arXiv:2012.08863, 2020. 





[41] Andrew Taylor, Andrew Singletary, Yisong Yue, and Aaron Ames. Learning for safety-critical control with control barrier functions. In Learning for Dynamics and Control, pages 708–717. PMLR, 2020. 





[17] Sophie Gruenbacher, Mathias Lechner, Ramin Hasani, Daniela Rus, Thomas A Henzinger, Scott Smolka, and Radu Grosu. Gotube: Scalable stochastic verification of continuous-depth models. arXiv preprint arXiv:2107.08467, 2021. 





[42] Keng Peng Tee, Shuzhi Sam Ge, and Eng Hock Tay. Barrier lyapunov functions for the control of outputconstrained nonlinear systems. Automatica, 45(4):918– 927, 2009. 





[18] Ramin Hasani. Interpretable Recurrent Neural Networks in Continuous-time Control Environments. PhD thesis, Technische Universitat Wien, 2020.¨ 





[43] Marin Toromanoff, Emilie Wirbel, Fred´ eric Wil-´ helm, Camilo Vejarano, Xavier Perrotton, and Fabien Moutarde. End to end vehicle lateral control using a single fisheye camera. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2018. 





[19] Ramin Hasani, Mathias Lechner, Alexander Amini, Daniela Rus, and Radu Grosu. A natural lottery ticket winner: Reinforcement learning with ordinary neural 





[44] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in neural information processing systems, pages 5998–6008, 2017. 





[45] Li Wang, Evangelos A Theodorou, and Magnus Egerstedt. Safe learning of quadrotor dynamics using barrier certificates. In 2018 IEEE International Conference on Robotics and Automation (ICRA), pages 2460–2465. IEEE, 2018. 





[46] Tsun-Hsuan Wang, Alexander Amini, Wilko Schwarting, Igor Gilitschenski, Sertac Karaman, and Daniela Rus. Learning interactive driving policies via data-driven simulation. arXiv preprint arXiv:2111.12137, 2021. 


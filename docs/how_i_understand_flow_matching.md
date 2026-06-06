# Video Summary: "How I Understand Flow Matching" (Jung-Bin Huang)

This document contains a structured summary of the mathematical formulas and concepts presented in Jung-Bin Huang's video tutorial [How I Understand Flow Matching](https://www.youtube.com/watch?v=DDq_pIfHqLs) (referenced in Scott H. Hawley's blog post as a primary guide for understanding continuous normalizing flows).

---

## 1. Discrete Time Normalizing Flows & Change of Variables

The goal is to map a simple base noise distribution $p_Z(z)$ (like a Gaussian) to a complex data distribution $p_X(x)$ via an invertible generator $x = g(z)$ where $z = g^{-1}(x)$.

### 1D Change of Variable `[02:04]`
To account for stretching or compressing local regions, the probability density must remain conserved:
$$p_X(x) = p_Z(z) \cdot \left| \frac{dz}{dx} \right|$$

### Higher-Dimensional Change of Variable `[03:12]`
Using the Jacobian Matrix ($J_g$) to account for volume changes in multiple dimensions:
$$p_X(x) = p_Z(z) \cdot \left| \det \left( \frac{\partial g^{-1}(x)}{\partial x} \right) \right| = p_Z(z) \cdot \left| \det J_g(z) \right|^{-1}$$

### Maximum Likelihood Estimation Objective `[03:27]`
Taking the log-likelihood of a sample under the model parameterization:
$$\log p_X(x) = \log p_Z(g^{-1}(x)) + \log \left| \det J_{g^{-1}}(x) \right|$$

### Composition of Flows (Stacking Layers) `[04:00]`
For a sequence of $K$ functions $g = g_K \circ \dots \circ g_1$, the log-likelihood simplifies to a summation of determinants across intermediate representations:
$$\log p_X(x) = \log p_Z(z_0) + \sum_{k=1}^K \log \left| \det J_{g_k^{-1}}(z_k) \right|$$

---

## 2. Simplifying the Jacobian (Special Architectures)

To make computing the determinant of the Jacobian ($\det J$) tractable in high dimensions, specialized neural network architectures are introduced.

### Coupling Layers `[04:15]`
Splits the input vector into two parts; the first part is copied directly, and the second part is scaled and shifted using functions of the first part:
$$x_{1:d} = z_{1:d}$$
$$x_{d+1:D} = z_{d+1:D} \odot \exp(s(z_{1:d})) + t(z_{1:d})$$
This creates a lower triangular Jacobian matrix where the determinant is simply the product of the diagonal elements:
$$\prod_{i=d+1}^{D} \exp(s(z_{1:d})_i)$$
`[05:25]`.

### Autoregressive Flows `[06:04]`
Each element of the output depends only on the preceding elements of the input:
$$x_i = \tau(z_i; h_i(z_{1:i-1}))$$

### Residual Flows `[07:47]`
Uses a residual connection $x = z + U(z)$. If $U$ is a contractive mapping, it can be inverted iteratively using a unique fixed point $z^*$:
$$z^* = x - U(z^*)$$
The determinant expands into an infinite series of matrix traces:
$$\log \left| \det (I + J_U) \right| = \sum_{k=1}^{\infty} \frac{(-1)^{k-1}}{k} \text{Tr}((J_U)^k)$$ `[08:08]`
The trace is estimated using Monte Carlo sampling via Hutchinson's trace estimator:
$$\text{Tr}(A) = \mathbb{E}_{v \sim \mathcal{N}(0, I)} [v^T A v]$$ `[08:46]`

---

## 3. Continuous Normalizing Flows (CNFs)

Instead of discrete layers, CNFs push samples through time continuously using an Ordinary Differential Equation (ODE).

### Neural ODE Formulation `[09:25]`
As the number of discrete residual layers approaches infinity ($K \to \infty$), the process becomes a continuous vector field $u_t$ parameterized by a neural network $\theta$:
$$\frac{dx_t}{dt} = u_t(x_t; \theta)$$

### The Continuity (Transport) Equation `[11:06]`
Describes the evolution of the probability density $p_t(x)$ over time given the vector field $u_t$:
$$\frac{\partial p_t(x)}{\partial t} + \nabla \cdot (p_t(x) u_t(x)) = 0$$
(Where $\nabla \cdot$ is the divergence operator).

---

## 4. Flow Matching

Because integrating Neural ODEs during training is computationally slow, Flow Matching introduces a regression loss to directly match a target vector field without solving the ODE.

### Unconditional Flow Matching Objective `[12:05]`
An $L_2$ regression loss to match the network $v_t$ with the true data-generating vector field $u_t$:
$$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}(0,1), x \sim p_t(x)} \left[ \| v_t(x; \theta) - u_t(x) \|^2 \right]$$
*Note: In practice, $u_t(x)$ and $p_t(x)$ are unknown, making this objective intractable directly.*

### Conditional Flow Matching (CFM) Objective `[13:27]`
By conditioning on a latent variable or endpoint data point $z$ (such as an endpoint $x_1$), the marginal path becomes a mixture of conditional paths, making the objective tractable:
$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, q(z), p_t(x|z)} \left[ \| v_t(x; \theta) - u_t(x|z) \|^2 \right]$$

#### Example 1: Conditional on a single target point ($z = x_1$) `[13:03]`
The conditional path and its corresponding vector field pushing toward $x_1$:
$$p_t(x|x_1) = \mathcal{N}(x; t x_1, (1 - t(1-\sigma^2))I)$$
$$u_t(x|x_1) = \frac{x_1 - x}{1 - t}$$

#### Example 2: Independent Coupling ($z = (x_0, x_1)$) `[13:50]`
Sampling starting noise $x_0$ and target image $x_1$, the path interpolates linearly (used in frameworks like Rectified Flow or Stochastic Interpolants):
$$\psi_t(x) = (1 - t)x_0 + t x_1$$
The conditional vector field becomes constant over time for that pair:
$$u_t(x | x_0, x_1) = x_1 - x_0$$ `[14:06]`

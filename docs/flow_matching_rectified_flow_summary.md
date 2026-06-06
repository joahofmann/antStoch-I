# Flow Matching and Rectified Flow Models: A Physical and Mathematical Summary

This document provides a mathematically intuitive summary of Scott H. Hawley's ICLR 2025 award-winning tutorial *"Flow With What You Know"* (companion to the paper/blog post of the same name). It bridges the physical principles of fluid mechanics and particle transport with generative machine learning (Flow Matching and Rectified Flow models).

---

## 1. What is a "Flow"? (The Physical Picture)

In physics, fluid motion can be described in two ways:
*   **Lagrangian View (Particle Tracking):** Traces individual trajectories of fluid particles. The position $\vec{r}(t)$ of a particle is governed by the ordinary differential equation (ODE):

    $$\frac{d\vec{r}}{dt} = \vec{v}(\vec{r}, t)$$

*   **Eulerian View (Field Description):** Defines a continuous **velocity vector field** $\vec{v}(\vec{x}, t)$ that describes the fluid velocity at any fixed point in space $\vec{x}$ and time $t$.

### Non-Crossing Streamlines and Invertibility
In a single-phase fluid flow, **streamlines (particle trajectories) never cross**. If they crossed at some space-time point $(\vec{x}, t)$, it would imply the fluid has two different velocity vectors at the exact same location and time, which is physically undefined.

*   **In Generative AI:** The non-crossing property guarantees that the transformation mapping from the starting noise distribution $\rho_0$ (at $t=0$) to the target data distribution $\rho_1$ (at $t=1$) is a **bijective and invertible mapping**.
*   **Contrast with Diffusion:** This is a major contrast to **diffusion models**, which rely on stochastic random walks (Brownian motion described by SDEs) that are inherently non-invertible.

---

## 2. How Flow Matching (FM) / Rectified Flow (RF) Works

The goal of generative flow models is to transport a simple distribution $\rho_0$ (at $t=0$, e.g., Gaussian noise) to a target data distribution $\rho_1$ (at $t=1$, e.g., images). We parameterized this velocity field $\vec{v}_\theta(\vec{x}, t)$ using a neural network.

### Step 1: Naive Straight-Line Trajectories (Lagrangian Setup)
To train the model, we begin with a simple assumption: particles move in straight lines at constant speeds between randomly paired start and end points.
1.  Sample a starting point $\mathbf{x}_0 \sim \rho_0$ and a target data point $\mathbf{x}_1 \sim \rho_1$.
2.  **Randomly pair** them and define a straight trajectory (linear interpolation / LERP):

    $$\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$$

3.  The constant velocity of this particle is:

    $$\vec{v}_{\text{target}} = \mathbf{x}_1 - \mathbf{x}_0$$

### Step 2: Training the Network (Eulerian Velocity Field)
Because of the random pairing, these straight trajectories cross each other extensively (creating a multi-valued velocity field). To resolve this, we train the network $\vec{v}_\theta(\vec{x}, t)$ using Mean Squared Error (MSE) to predict the velocity of these trajectories:

$$\mathcal{L}(\theta) = \mathbb{E}_{t \sim U(0,1), \, \mathbf{x}_0 \sim \rho_0, \, \mathbf{x}_1 \sim \rho_1} \left[ \left\| \vec{v}_\theta(\mathbf{x}_t, t) - (\mathbf{x}_1 - \mathbf{x}_0) \right\|_2^2 \right]$$

### The Physics Analogy: "Seeking the Mean"
When multiple crossing straight-line trajectories pass through the same point $\vec{x}$ at time $t$, the network cannot output multiple velocities. Minimizing the quadratic loss forces the network to output the **expected velocity** of all particles passing through that point at that time:

$$\vec{v}_\theta(\vec{x}, t) = \mathbb{E} \left[ \mathbf{x}_1 - \mathbf{x}_0 \mid \mathbf{x}_t = \vec{x} \right]$$

This is physically analogous to how chaotic molecular collisions (Brownian motion) average out to produce a smooth, continuous, and non-crossing macroscopic fluid flow. Thus, the integrated paths of the learned field $\vec{v}_\theta$ **never cross** and are smooth, even though the training trajectories crossed.

---

## 3. "Reflow" to Go Straighter & Faster

Although the learned trajectories from Flow Matching do not cross, they are **highly curved**. 
*   Integrating highly curved paths using a simple numerical method like Forward Euler requires many small steps (often 50–100 evaluations) to prevent the particles from drifting off the paths.
*   To solve this, the *Rectified Flow* paper introduces **Reflow** (which acts as a distillation process).

```
[ Random Pairing: x0 & x1 ]  --->  [ Train Flow Matching Model 1 ]  ---> [ Curved Streamlines ]
                                                                                   |
                                                                                   v
[ Straight Streamlines (Fast) ] <--- [ Train Reflow Model 2 ] <--- [ Pair x0 & x1_sim (Endpoints) ]
```

### The Reflow Algorithm:
1.  Take a source point $\mathbf{x}_0 \sim \rho_0$.
2.  Use the trained *teacher* model $\vec{v}_\theta$ to integrate $\mathbf{x}_0$ all the way to $t=1$, yielding a **simulated target point** $\mathbf{x}_1^{\text{sim}}$.
3.  Form a new dataset of paired points $(\mathbf{x}_0, \mathbf{x}_1^{\text{sim}})$. Because $\mathbf{x}_1^{\text{sim}}$ is the actual endpoint of the streamline starting at $\mathbf{x}_0$, these points are **perfectly aligned** along the natural flow.
4.  Train a new *student* model $\vec{v}_\phi$ using these new straight-line trajectories:

    $$\mathbf{x}_t^{\text{rect}} = (1-t)\mathbf{x}_0 + t\mathbf{x}_1^{\text{sim}}$$

    $$\mathcal{L}(\phi) = \mathbb{E} \left[ \left\| \vec{v}_\phi(\mathbf{x}_t^{\text{rect}}, t) - (\mathbf{x}_1^{\text{sim}} - \mathbf{x}_0) \right\|_2^2 \right]$$

### Result:
The trajectories of the student model $\vec{v}_\phi$ are **straight lines**. Because they are straight and do not cross, we can generate samples in as few as **1 or 2 steps** using simple Euler integration without accumulating error.

---

## 4. Upgrading Integration (No Retraining Needed)

To get highly accurate simulated targets $\mathbf{x}_1^{\text{sim}}$ for Reflow, two engineering upgrades are applied to the ODE integration:

### 1. Time Warping
Trajectories are often straight at the boundaries ($t \approx 0, 1$) but highly curved in the middle ($t \approx 0.5$). We want to take smaller steps where the path is curved. We map uniform steps to non-uniform steps using an S-shaped polynomial warping function:

$$f(t) =  4(1-s)t^3 + 6(s-1) t^2 + (3-2s)t, \quad t\in[0,1], \quad s\in[0,3/2]$$

*   The parameter $s$ controls the slope at $t=0.5$. Setting $s=0.5$ concentrates more integration steps in the middle of the trajectory, significantly increasing accuracy. (This is equivalent to the sampling schedules used in Stable Diffusion 3 and FLUX).

### 2. High-Order ODE Solvers (RK4)
Rather than the standard Forward Euler method, we use the **4th-order Runge-Kutta (RK4)** scheme. Per step, RK4 requires 4 function evaluations, but its high-order accuracy allows for much larger step sizes, making the total generation both faster and significantly more accurate.

---

## 5. Connections to Other Generative Models

| Model Class | Dynamics | Kinematic / Physical Analogy | Key Difference |
| :--- | :--- | :--- | :--- |
| **Diffusion Models** | Stochastic SDE | **Brownian Motion:** Particles undergo constant random jitter. | Non-invertible; requires noise schedules. |
| **Flow Matching / RF** | Deterministic ODE | **Macroscopic Fluid Flow:** Smooth streamlines; bulk transport. | Invertible; paths can be straightened (Reflow). |
| **Normalizing Flows** | Deterministic ODE | **Hamiltonian Mechanics:** Volume/probability is strictly conserved (Liouville's Theorem). | Requires calculating Jacobian determinants, limiting model capacity. |
| **Optimal Transport** | Deterministic ODE | **Geodesics:** Straight paths moving at constant speeds. | Reflow naturally converges to Optimal Transport paths (Wasserstein geodesics). |

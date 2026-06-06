# Stochastic Interpolants & Workspace Setup Summary

This document summarizes the theoretical foundations of **Stochastic Interpolants** and the project setup completed in this conversation session.

---

## 1. Theoretical Foundations of Stochastic Interpolants

A **Stochastic Interpolant** is a mathematical framework that unifies **Normalizing Flows (ODEs)** and **Diffusion Models (SDEs)**. It defines a continuous-time stochastic process $\{x_t\}_{t \in [0,1]}$ that directly connects a source distribution $\rho_0$ (e.g., noise) to a target distribution $\rho_1$ (e.g., real data) in finite time.

### Mathematical Formulation
The stochastic interpolant process is defined as:

$$x_t = \alpha(t) x_0 + \beta(t) x_1 + \gamma(t) z$$

Where:
*   **$x_0 \sim \rho_0$** is a sample from the source distribution (e.g., base Gaussian noise).
*   **$x_1 \sim \rho_1$** is a sample from the target distribution (e.g., real images).
*   **$z \sim \mathcal{N}(0, \mathbf{I})$** is an independent standard Gaussian variable representing stochastic fluctuations (the noise bridge).
*   **$\alpha(t), \beta(t), \gamma(t)$** are smooth time-dependent scaling functions on $t \in [0, 1]$.

#### Boundary Conditions:
To ensure the process begins exactly at $x_0$ and ends exactly at $x_1$:
*   **At $t = 0$:** $\alpha(0) = 1$, $\beta(0) = 0$, and $\gamma(0) = 0$ $\implies x_0 = x_0$
*   **At $t = 1$:** $\alpha(1) = 0$, $\beta(1) = 1$, and $\gamma(1) = 0$ $\implies x_1 = x_1$

#### Common Configurations:
1.  **Deterministic Flows (Flow Matching / Rectified Flow):**
    We set the noise bridge $\gamma(t) = 0$, with linear schedules $\alpha(t) = 1 - t$ and $\beta(t) = t$. The path is a straight-line interpolation:
    $$x_t = (1 - t)x_0 + t x_1$$
2.  **Stochastic Bridges (Diffusion):**
    We set $\gamma(t) > 0$ for $t \in (0, 1)$ to add random noise along the trajectory. A typical choice is $\gamma(t) = \sqrt{t(1 - t)}$.

---

## 2. SDE vs. ODE in Generative Modeling

Stochastic Interpolants allow us to generate samples using either stochastic or deterministic dynamics:

| Metric | SDE (Stochastic Differential Equation) | ODE (Ordinary Differential Equation) |
| :--- | :--- | :--- |
| **Equation** | $dx_t = b(t, x_t) dt + g(t) dW_t$ | $dx_t = v(t, x_t) dt$ |
| **Mechanism** | Uses a deterministic drift $b(t, x_t)$ and a random noise diffusion $g(t) dW_t$. | Integration along a purely deterministic vector field $v(t, x_t)$. |
| **Characteristics** | Stochastic paths. Noise added at each step helps correct errors, but requires many evaluation steps ($50 - 1000$ NFE). | Deterministic paths. If paths are straightened (Rectified Flow), we can jump from noise to data in a **single step** ($NFE = 1$). |
| **Examples** | DDPM, SGM (Score-based Generative Models) | Flow Matching, Rectified Flow |

---

## 3. DDPM: Score-Based View vs. ELBO View

The training of traditional diffusion models (like DDPM) is approached from two mathematical perspectives:

*   **The Score-Based View (Denoising Score Matching):**
    We train a network $s_\theta(x_t, t)$ to predict the score function $\nabla_{x_t} \log p_t(x_t)$ of the noised data distribution. Since the noising process is Gaussian, the score is proportional to the added noise:
    $$\nabla_{x_t} \log q(x_t|x_0) = -\frac{\epsilon}{\sqrt{1 - \bar{\alpha}_t}}$$
    Thus, learning the score is equivalent to predicting the noise vector $\epsilon$.
*   **The ELBO View (Variational Inference):**
    Formulates diffusion as a Hierarchical Variational Autoencoder (HVAE) and maximizes the Evidence Lower Bound (ELBO). The loss decomposes into KL-divergence terms between the true reverse transitions and the parameterized model. Because both are Gaussian, the KL divergence simplifies to the Mean Squared Error (MSE) between the actual noise $\epsilon$ and the predicted noise.

Both views yield the **simplified DDPM loss**:
$$\mathcal{L}_{\text{simple}}(\theta) = \mathbb{E}_{t, x_0, \epsilon} \left[ \|\epsilon - \epsilon_\theta(x_t, t)\|^2 \right]$$

---

## 4. Workspace Directory Structure Sketch

The active folder layout is organized as follows:

```text
C:\Users\joach\
├── antiG\                                    <-- MAIN PROJECT ROOT
│   │
│   ├── antiCeleb064\                         <-- CelebA 64x64 Face Experiments
│   │   ├── celeb064_resalts\                 <-- Model weights & checkpoints
│   │   └── FM_RM_Colab_ZBook01.ipynb
│   │
│   ├── compDDPM-FM\                          <-- Core Python Codebase
│   │   └── models.py / ddpm.py / flow_matching.py
│   │
│   ├── report\                               
│   │   └── rm_fm_ddpm_report.md              <-- Markdown Comparison Report
│   │
│   └── antStoch-I\                           <-- ACTIVE WORKSPACE (THIS PROJECT)
│       ├── .git/                             <-- Local Git Repo (Initialized)
│       ├── .gitignore                        <-- Python Git Ignore Config
│       ├── README.md                         <-- Project Introduction
│       └── docs/
│           └── stochastic_interpolants_summary.md <-- This Summary File
```

---

## 5. Git & Workspace Initialization

The following steps were executed to set up the repository:
1.  **Git Initialization**: Run `git init` inside `antStoch-I`.
2.  **Configured Local Identity**:
    ```bash
    git config user.name "Joachim"
    git config user.email "joachim@example.com"
    ```
3.  **Staged and Committed Files**:
    ```bash
    git add .
    git commit -m "Initial commit"
    ```

### To Link and Push to GitHub:
To upload this repository to your GitHub account:
1.  Create a blank repository named `antStoch-I` on GitHub (do **not** check the box to initialize with a README).
2.  Run the following commands in the `antStoch-I` folder to link and push:
    ```bash
    git remote add origin <your-copied-github-repo-url>
    git branch -M main
    git push -u origin main
    ```

# Derivation: The Jacobian of Coupling Layers

In high-dimensional Normalizing Flows (e.g. generating an image with $D$ pixels where $D = 196,608$), calculating the determinant of a general $D \\times D$ Jacobian matrix takes **$\\mathcal{O}(D^3)$ time** (via LU decomposition). This is computationally impossible to do per training step.

**Coupling Layers** are designed to reduce this complexity to **$\\mathcal{O}(D)$ time** (linear time) by forcing the Jacobian to be block lower triangular.

---

## Step 1: Splitting the Vector Space
We split the input vector $z \\in \\mathbb{R}^D$ and output vector $x \\in \\mathbb{R}^D$ into two parts:
*   **Part A (First $d$ elements):** $z_A = z_{1:d}$ and $x_A = x_{1:d}$
*   **Part B (Remaining $D-d$ elements):** $z_B = z_{d+1:D}$ and $x_B = x_{d+1:D}$

The coupling layer transforms these parts using two arbitrary neural networks, $s(z_A)$ (scale) and $t(z_A)$ (translation):
1.  **Identity Mapping:** 
    $$x_A = z_A$$
2.  **Affine Transformation:** 
    $$x_B = z_B \\odot \\exp(s(z_A)) + t(z_A)$$

*(where $\\odot$ represents the Hadamard / element-wise product).*

---

## Step 2: The Block Jacobian Matrix
The Jacobian matrix $J$ of the transformation $x = g(z)$ is the matrix of all first-order partial derivatives. We can write $J$ in block form as:

$$J = \\frac{\\partial(x_A, x_B)}{\\partial(z_A, z_B)} = \\begin{pmatrix}
\\frac{\\partial x_A}{\\partial z_A} & \\frac{\\partial x_A}{\\partial z_B} \\\\
\\frac{\\partial x_B}{\\partial z_A} & \\frac{\\partial x_B}{\\partial z_B}
\\end{pmatrix}$$

Let us compute each of these four blocks:

### Block 1: $\\frac{\\partial x_A}{\\partial z_A}$
Since $x_A = z_A$, each element $x_i$ (for $i \\le d$) depends only on $z_i$. Thus, the derivative is the Identity matrix:
$$\\frac{\\partial x_A}{\\partial z_A} = \\mathbf{I}_{d \\times d}$$

### Block 2: $\\frac{\\partial x_A}{\\partial z_B}$
Since $x_A = z_A$, the first part of the output does not depend on the second part of the input $z_B$ at all. Thus, all these derivatives are zero:
$$\\frac{\\partial x_A}{\\partial z_B} = \\mathbf{0}_{d \\times (D-d)}$$

### Block 3: $\\frac{\\partial x_B}{\\partial z_A}$
The second part of the output $x_B$ depends on $z_A$ through the neural networks $s(z_A)$ and $t(z_A)$. This creates a complicated matrix of derivatives:
$$\\frac{\\partial x_B}{\\partial z_A} = \\mathbf{M}_{(D-d) \\times d}$$

### Block 4: $\\frac{\\partial x_B}{\\partial z_B}$
Let us look at a single element $x_i$ in $x_B$ (where $i \\in \\{d+1, \\dots, D\\}$):
$$x_i = z_i \\cdot \\exp(s(z_A)_{i-d}) + t(z_A)_{i-d}$$
Now we take the partial derivative with respect to $z_j$ (where $j \\in \\{d+1, \\dots, D\\}$):
*   If $i \\neq j$: $\\frac{\\partial x_i}{\\partial z_j} = 0$ (because $x_i$ only depends on its own $z_i$ element).
*   If $i = j$: $\\frac{\\partial x_i}{\\partial z_i} = \\exp(s(z_A)_{i-d})$ (the coefficient of $z_i$).

Because all off-diagonal terms are $0$, this block is a **diagonal matrix**:
$$\\frac{\\partial x_B}{\\partial z_B} = \\text{diag}\\left( \\exp(s(z_A)) \\right)$$

---

## Step 3: Calculating the Determinant

Now, let us substitute these blocks back into our full Jacobian matrix $J$:

$$J = \\begin{pmatrix}
\\mathbf{I} & \\mathbf{0} \\\\
\\mathbf{M} & \\text{diag}\\left(\\exp(s(z_A))\\right)
\\end{pmatrix}$$

This is a **block lower triangular matrix** (since the top-right block is a zero matrix). 

From linear algebra, the determinant of any block triangular matrix is simply the product of the determinants of its diagonal blocks:
$$\\det(J) = \\det(\\mathbf{I}) \\cdot \\det\\left( \\text{diag}\\left(\\exp(s(z_A))\\right) \\right)$$

1.  The determinant of the identity matrix is $1$: $\\det(\\mathbf{I}) = 1$.
2.  The determinant of a diagonal matrix is simply the product of its diagonal elements:
    $$\\det\\left( \\text{diag}\\left(\\exp(s(z_A))\\right) \\right) = \\prod_{i=d+1}^{D} \\exp(s(z_A)_{i-d})$$

Therefore:

$$\\det(J) = \\prod_{i=d+1}^{D} \\exp\\left(s(z_{1:d})_{i-d}\\right)$$

---

## Why is this so powerful?

1.  **Linear Complexity $\\mathcal{O}(D)$:** We only have to multiply $D-d$ numbers. There is no matrix inversion or complex decomposition needed.
2.  **Arbitrary Neural Networks:** The matrix $\\mathbf{M}$ (which contains the complex partial derivatives of $s(z_A)$ and $t(z_A)$) is in the lower-left block. Because of the triangular structure, **it has no effect on the determinant**. Thus, $s$ and $t$ can be arbitrarily deep, non-linear, non-invertible neural networks (e.g. ResNets, transformers), and the layer remains mathematically invertible with a simple Jacobian determinant!
3.  **Inversion is Trivial:** To go backwards ($x \\to z$):
    *   $z_A = x_A$
    *   $z_B = (x_B - t(x_A)) \\odot \\exp(-s(x_A))$
    *(which uses the same functions $s$ and $t$, requiring no neural network inversion).*

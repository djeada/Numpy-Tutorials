# Matrices: transformations, products, and numerical size

A real matrix $A\in\mathbb R^{m\times n}$ maps $n$-component input vectors to
$m$-component output vectors. The formula

$$
(Ax)_i=\sum_{j=0}^{n-1}A_{ij}x_j
$$

says that each output is a weighted sum of the inputs. An array with more than
two axes can hold a batch of matrices; those extra axes require explicit care.

## Elementwise multiplication and matrix multiplication answer different questions

For $A\in\mathbb R^{m\times k}$ and $B\in\mathbb R^{k\times n}$,

$$
(AB)_{ij}=\sum_{\ell=0}^{k-1}A_{i\ell}B_{\ell j}.
$$

The shared dimension $k$ is summed out. Elementwise multiplication `A * B`
instead multiplies corresponding entries, subject to broadcasting.

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[2, 0], [1, 2]])
x = np.array([1, 2])

print((A @ x).tolist())
print((A * B).tolist())
print((A @ B).tolist())
print((B @ A).tolist())
```

```text
[5, 11]
[[2, 0], [3, 8]]
[[4, 4], [10, 8]]
[[2, 4], [7, 10]]
```

For example, $(AB)_{1,0}=3\cdot2+4\cdot1=10$.
Both products happen to have shape `(2,2)`, so shape alone cannot establish
that the operation matches the intended mathematics.

Matrix multiplication is associative when shapes permit it, but generally not
commutative. In $RSx$, $S$ acts first.

```python
import numpy as np

rotation = np.array([[0, -1], [1, 0]])  # 90 degrees counterclockwise
stretch = np.diag([2, 1])               # double the x coordinate
x = np.array([1, 1])

print((rotation @ stretch @ x).tolist())
print((stretch @ rotation @ x).tolist())
np.testing.assert_array_equal(
    rotation @ (stretch @ x), (rotation @ stretch) @ x
)
```

```text
[-1, 2]
[-2, 1]
```

The first path is $(1,1)\mapsto(2,1)\mapsto(-1,2)$.
The second is $(1,1)\mapsto(-1,1)\mapsto(-2,1)$.

## Batched `@` broadcasts leading axes; `dot` can create extra axes

For `@`, the last two axes hold matrix rows and columns:

$$
(\ldots,m,k)\ @\ (\ldots,k,n)\ \longrightarrow\ (\ldots,m,n).
$$

The leading axes broadcast. In contrast, `np.dot` on higher-dimensional inputs
contracts the first input's last axis with the second input's second-to-last
axis and retains the other axes from both inputs.

```python
import numpy as np

A = np.arange(24).reshape(2, 3, 4)  # two 3-by-4 matrices
B = np.arange(40).reshape(2, 4, 5)  # two 4-by-5 matrices

paired = A @ B
all_batches = np.dot(A, B)
print(paired.shape)
print(all_batches.shape)
print(np.array_equal(paired, np.einsum("bik,bkj->bij", A, B)))
print(A.T.shape)
print(A.swapaxes(-1, -2).shape)
```

```text
(2, 3, 5)
(2, 3, 2, 5)
True
(4, 3, 2)
(2, 4, 3)
```

`paired[b]` is `A[b] @ B[b]`. `all_batches[b, :, c, :]` is `A[b] @ B[c]`:
`dot` retained both batch indices. For two-dimensional arrays, `dot` and `@`
give the same product; this equivalence does not generalize to batches.

Similarly, `A.T` reverses **all** axes. To transpose each matrix in a batch,
swap only the last two axes. For complex adjoints, also conjugate:
`A.conj().swapaxes(-1, -2)`.

## Matrix norms measure different kinds of size

For a matrix $A$, common choices are

$$
\|A\|_F=\sqrt{\sum_{i,j}|A_{ij}|^2},\qquad
\|A\|_1=\max_j\sum_i|A_{ij}|,\qquad
\|A\|_\infty=\max_i\sum_j|A_{ij}|.
$$

The induced Euclidean norm is

$$
\|A\|_2=\max_{\|x\|_2=1}\|Ax\|_2=\sigma_{\max}(A).
$$

It measures the largest amplification of a unit input vector.
The default `np.linalg.norm(A)` for a two-dimensional matrix is the Frobenius
norm, not the induced $2$-norm.

```python
import numpy as np

A = np.array([[1.0, 2.0], [3.0, 4.0]])
norms = [np.linalg.norm(A), np.linalg.norm(A, 2),
         np.linalg.norm(A, 1), np.linalg.norm(A, np.inf)]
print(np.round(norms, 6).tolist())
print(float(np.abs(A).max()))

B = np.array([[2.0, 0.0], [1.0, 2.0]])
print(bool(np.linalg.norm(A @ B, 2)
           <= np.linalg.norm(A, 2) * np.linalg.norm(B, 2) + 1e-12))
```

```text
[5.477226, 5.464986, 6.0, 7.0]
4.0
True
```

Here $\|A\|_F=\sqrt{30}$, the largest absolute column sum is $6$, and the largest
absolute row sum is $7$. The largest absolute entry, $4$, is a different quantity.
For vectors, `ord=np.inf` means the largest absolute component; for matrices it
means the largest absolute row sum.

The induced norm gives the error bound

$$
\|Ax-Ay\|_2\leq\|A\|_2\|x-y\|_2.
$$

Its submultiplicative property,
$\|AB\|_2\leq\|A\|_2\|B\|_2$, explains why norm products can bound errors in
composed transformations. Roundoff can affect checks at the inequality boundary,
so the numerical check above includes a small tolerance.

## A small determinant is not a scale-independent singularity test

For a square matrix, a nonzero determinant is an exact-arithmetic criterion for
invertibility. Its magnitude changes under scaling:
$\det(cA)=c^n\det(A)$ for an $n\times n$ matrix.
The condition number $\kappa_2(A)=\sigma_{\max}/\sigma_{\min}$ instead measures
relative sensitivity.

```python
import numpy as np

small = 1e-8 * np.eye(2)
sensitive = np.diag([1e8, 1e-8])

print(f"{np.linalg.det(small):.0e}", f"{np.linalg.cond(small):.0f}")
print(f"{np.linalg.det(sensitive):.0e}", f"{np.linalg.cond(sensitive):.0e}")

dependent = np.array([[1.0, 2.0], [2.0, 4.0]])
print(np.linalg.matrix_rank(dependent))
```

```text
1e-16 1
1e+00 1e+16
1
```

The first matrix has a tiny determinant but condition number $1$.
The second has determinant $1$ but amplifies some relative perturbations severely.
Use singular values or a condition number to investigate sensitivity.

`matrix_rank` estimates rank using a singular-value tolerance. It is a numerical
decision about a threshold, not symbolic proof that a tiny singular value is
exactly zero. Also distinguish matrix rank from `ndim`: `dependent` has two
axes but rank one.

## Solving a system does not require constructing an inverse

The identity $A^{-1}A=I$ explains the formal solution $x=A^{-1}b$.
Numerically, ask for the action you need: `np.linalg.solve(A, b)`.
Constructing the entire inverse adds work when the goal is only to solve a system.

```python
import numpy as np

A = np.array([[2.0, 1.0], [1.0, 3.0]])
b = np.array([1.0, 2.0])
x = np.linalg.solve(A, b)

print(np.round(x, 6).tolist())
print(np.allclose(A @ x, b, rtol=1e-12, atol=1e-12))
print(np.allclose(A @ np.eye(2), A))
```

```text
[0.2, 0.6]
True
True
```

The substitution check is $2(0.2)+0.6=1$ and $0.2+3(0.6)=2$.
`solve` requires a square, full-rank coefficient matrix. Least squares and
rank-deficient problems need different treatment.

## Singular values explain low-rank structure and principal directions

For a real $m\times n$ matrix, the reduced SVD is

$$
A=U\operatorname{diag}(s)V^\top,\qquad
U\in\mathbb R^{m\times k},\ s\in\mathbb R^k,\ V^\top\in\mathbb R^{k\times n},
\quad k=\min(m,n).
$$

The singular values are nonnegative and sorted largest first. Keeping only the
first term $s_0u_0v_0^\top$ gives a rank-one approximation. More generally,
truncating the SVD gives an optimal approximation of the chosen rank in the
Frobenius and spectral norms.

```python
import numpy as np

X = np.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])  # already centered
U, s, Vh = np.linalg.svd(X, full_matrices=False)
rank_one = s[0] * np.outer(U[:, 0], Vh[0])

covariance = X.T @ X / (len(X) - 1)
eigenvalues, directions = np.linalg.eigh(covariance)
print(U.shape, s.shape, Vh.shape)
print(np.round(s, 6).tolist())
print(np.allclose((U * s) @ Vh, X))
print(np.allclose(rank_one, X))
print(covariance.tolist())
print(np.round(eigenvalues, 6).tolist())
print(bool(np.isclose(abs(directions[:, -1] @ Vh[0]), 1)))
```

```text
(3, 2) (2,) (2, 2)
[2.0, 0.0]
True
True
[[1.0, 1.0], [1.0, 1.0]]
[0.0, 2.0]
True
```

The observations lie on one line, so one singular direction reconstructs them.
For centered observations in rows, covariance is $X^\top X/(n-1)$, and its
eigenvalues are $s_i^2/(n-1)$. The principal direction is proportional to $(1,1)$.
Its sign is arbitrary: $v$ and $-v$ describe the same line, so comparing exact
printed eigenvectors is a fragile check. `eigh` is appropriate here because
covariance is symmetric; it returns eigenvalues in ascending order.

## A matrix identity can save memory and lose accuracy

The squared distance identity

$$
\|x_i-x_j\|_2^2=\|x_i\|_2^2+\|x_j\|_2^2-2x_i^\top x_j
$$

avoids an $n\times n\times d$ difference array.

```python
import numpy as np

X = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
squared_length = np.sum(X * X, axis=1)
squared_distance = (squared_length[:, None] + squared_length[None, :]
                    - 2 * (X @ X.T))
distance = np.sqrt(np.maximum(squared_distance, 0.0))
print(distance.tolist())

# Nearly equal vectors with a large common offset expose cancellation.
Y = np.array([[1e8, 1e8], [1e8 + 1, 1e8]])
q = np.sum(Y * Y, axis=1)
via_products = q[:, None] + q[None, :] - 2 * (Y @ Y.T)
print(float(np.sum((Y[0] - Y[1]) ** 2)))
print(float(via_products[0, 1]))
```

```text
[[0.0, 3.0, 4.0], [3.0, 0.0, 5.0], [4.0, 5.0, 0.0]]
1.0
0.0
```

In the second calculation, subtracting quantities near $4\cdot10^{16}$ loses
the squared distance $1$. Clipping negative roundoff before a square root cannot
recover lost positive distances. For close points with large offsets, consider
subtracting a common origin first or computing direct differences in blocks.
The $n\times n$ output itself still costs quadratic memory.

See NumPy's [matrix multiplication](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html),
[SVD](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html), and
[condition number](https://numpy.org/doc/stable/reference/generated/numpy.linalg.cond.html) references.

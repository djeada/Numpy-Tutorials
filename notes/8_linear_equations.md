# Linear equations: exact solutions, least squares, and sensitivity

For $A\in\mathbb R^{m\times n}$, $x\in\mathbb R^n$, and $b\in\mathbb R^m$,

$$
Ax=b,\qquad \sum_{j=0}^{n-1}A_{ij}x_j=b_i
$$

represents $m$ equations in $n$ unknowns. Three separate questions matter:
does an exact solution exist, is it unique, and how sensitive is it to small
changes in the inputs?

## Choose the problem before choosing a solver

In exact arithmetic, consistency and uniqueness can be described by rank:

| Condition | Exact solutions |
| --- | --- |
| $\operatorname{rank}([A\mid b])>\operatorname{rank}(A)$ | None |
| $\operatorname{rank}([A\mid b])=\operatorname{rank}(A)=n$ | One |
| $\operatorname{rank}([A\mid b])=\operatorname{rank}(A)<n$ | Infinitely many |

Here $[A\mid b]$ is the matrix formed by appending $b$ as a column.
A rectangular system can have a unique exact solution when it is consistent and
has full column rank. That does not make it an input for `np.linalg.solve`,
which requires a square, full-rank matrix.

In floating-point work, rank depends on a singular-value tolerance. Comparing
two computed ranks is not a universal replacement for examining residuals,
data accuracy, and singular values.

## Solve a square system and substitute the answer

Consider

$$
\begin{aligned}
2x_0+5x_2&=10,\\
3x_0+4x_1+8x_2&=15,\\
2x_0+7x_1+3x_2&=5.
\end{aligned}
$$

The exact solution is $(15/23,-5/23,40/23)$.

```python
import numpy as np

A = np.array([[2.0, 0.0, 5.0],
              [3.0, 4.0, 8.0],
              [2.0, 7.0, 3.0]])
b = np.array([10.0, 15.0, 5.0])
x = np.linalg.solve(A, b)
residual = b - A @ x

print(np.round(x, 6).tolist())
print(bool(np.linalg.norm(residual) < 1e-12))
np.testing.assert_allclose(x, np.array([15, -5, 40]) / 23)
```

```text
[0.652174, -0.217391, 1.73913]
True
```

The first equation checks as $2(15/23)+5(40/23)=230/23=10$.
Use `solve(A, b)` for this task instead of explicitly computing `inv(A) @ b`.
A solver returning without an exception does not establish that an
ill-conditioned answer is accurate.

## Right-hand-side shape can corrupt a residual check

For a single system, `b.shape == (m,)` produces a vector solution of shape
`(n,)`. A right-hand side of shape `(m,1)` produces a column matrix.
Mixing the two representations can create an unintended matrix through broadcasting.

```python
import numpy as np

A = np.array([[2.0, 1.0], [1.0, 3.0]])
b = np.array([1.0, 2.0])
x = np.linalg.solve(A, b)

print(x.shape)
print((A @ x - b).shape)
print((A @ x - b[:, None]).shape)
print(np.allclose(A @ x, b[:, None]))

# Columns of B are separate right-hand sides for the same A.
B = np.column_stack((b, 2 * b))
solutions = np.linalg.solve(A, B)
print(np.round(solutions, 6).tolist())
print(np.allclose(A @ solutions, B))
```

```text
(2,)
(2,)
(2, 2)
False
[[0.2, 0.4], [0.6, 1.2]]
True
```

The incorrect subtraction forms all pairwise differences between the predicted
entries and the column entries. Check that the predicted and target shapes agree
before interpreting a residual norm.

## Singular systems: least squares does not mean an exact solution exists

For

$$
A=\begin{bmatrix}1&1\\2&2\end{bmatrix},
$$

$b=(2,4)$ is consistent and leaves the line $x_0+x_1=2$ of solutions.
With $b=(2,5)$, the two equations contradict one another.
`lstsq` still returns an answer: it minimizes $\|Ax-b\|_2$.

```python
import numpy as np

A = np.array([[1.0, 1.0], [2.0, 2.0]])

for b in (np.array([2.0, 4.0]), np.array([2.0, 5.0])):
    x, reported_residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    r = b - A @ x
    print(np.round(x, 6).tolist(), round(float(r @ r), 6))
    print(int(rank), reported_residuals.size)
```

```text
[1.0, 1.0] 0.0
1 0
[1.2, 1.2] 0.2
1 0
```

The second answer predicts $(2.4,4.8)$ and has residual $(-0.4,0.2)$, giving
$0.4^2+0.2^2=0.2$. An empty `reported_residuals` array does **not** mean zero
error. NumPy returns an empty array there when the matrix lacks full column
rank or when $m\leq n$. Compute `b - A @ x` yourself when you need the residual.

If several vectors attain the minimum residual, `lstsq` selects the one with
smallest Euclidean norm. For the consistent case, this selects $(1,1)$ among
all vectors whose components sum to $2$.

## Fit a line by least squares, with the numbers worked through

Fit $y_i\approx\beta_0+\beta_1t_i$ to
$(t,y)=(0,1),(1,2),(2,2)$. The design matrix contains a column of ones and
a column of times:

$$
M=\begin{bmatrix}1&0\\1&1\\1&2\end{bmatrix}.
$$

The least-squares objective is

$$
\min_\beta\|M\beta-y\|_2^2.
$$

The fitted line has $\beta=(7/6,1/2)$, predictions $(7/6,5/3,13/6)$,
and residual $(-1/6,1/3,-1/6)$.

```python
import numpy as np

t = np.array([0.0, 1.0, 2.0])
y = np.array([1.0, 2.0, 2.0])
M = np.column_stack((np.ones_like(t), t))
beta, residuals, rank, s = np.linalg.lstsq(M, y, rcond=None)
prediction = M @ beta
r = y - prediction

print(np.round(beta, 6).tolist())
print(np.round(prediction, 6).tolist())
print(np.round(r, 6).tolist())
print(round(float(r @ r), 6))
print(np.allclose(M.T @ r, 0, rtol=0, atol=1e-12))
np.testing.assert_allclose(residuals, [r @ r])

# For full column rank, reduced QR gives another derivation of the solution.
Q, R = np.linalg.qr(M, mode="reduced")
beta_qr = np.linalg.solve(R, Q.T @ y)
print(np.allclose(beta_qr, beta))
```

```text
[1.166667, 0.5]
[1.166667, 1.666667, 2.166667]
[-0.166667, 0.333333, -0.166667]
0.166667
True
True
```

The residual is perpendicular to both columns of $M$: $M^\top r=0$.
The residual sum of squares is $1/36+1/9+1/36=1/6$.

For full column rank, $M=QR$ with $Q^\top Q=I$, so the minimizer satisfies
$R\beta=Q^\top y$. This QR recipe requires nonsingular $R$.
For rank-deficient inputs, use `lstsq` rather than blindly solving that
triangular system.

The normal equations $M^\top M\beta=M^\top y$ are useful for deriving the
orthogonality condition, but forming $M^\top M$ squares the $2$-norm condition
number when $M$ has full column rank. Prefer `lstsq` to implementing the
normal equations as a general numerical solver.

## Underdetermined systems need a rule for choosing among solutions

For

$$
A=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix},\qquad
b=\begin{bmatrix}7\\8\end{bmatrix},
$$

there are more unknowns than independent equations. The minimum-norm solution is

$$
x_*=\left(-\frac{55}{18},\frac19,\frac{59}{18}\right).
$$

```python
import numpy as np

A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = np.array([7.0, 8.0])
x = np.linalg.pinv(A) @ b
x_lstsq, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
null_direction = np.array([1.0, -2.0, 1.0])

print(np.round(x, 6).tolist())
print(np.allclose(A @ x, b))
print(np.allclose(A @ null_direction, 0))
print(bool(np.isclose(x @ null_direction, 0, rtol=0, atol=1e-12)))
print(bool(np.linalg.norm(x + null_direction) > np.linalg.norm(x)))
np.testing.assert_allclose(x, x_lstsq)
```

```text
[-3.055556, 0.111111, 3.277778]
True
True
True
True
```

Every vector $x_*+tz$, with $z=(1,-2,1)$, solves the same equations because
$Az=0$. The minimum-norm solution is orthogonal to this null direction:

$$
\|x_*+tz\|_2^2=\|x_*\|_2^2+t^2\|z\|_2^2.
$$

Thus $t=0$ minimizes the length. The pseudoinverse encodes this particular choice;
it does not establish that the physical problem itself has only one solution.

## SVD inversion must discard numerically zero directions

For the reduced SVD $A=U\operatorname{diag}(s)V^\top$,

$$
A^+=V\operatorname{diag}(s^+)U^\top,\qquad
s_i^+=
\begin{cases}
1/s_i,&s_i>\tau,\\
0,&s_i\leq\tau.
\end{cases}
$$

Blindly forming `1 / s` can divide by zero or amplify roundoff in a nearly
null direction. A relative cutoff uses $\tau=rcond\cdot s_{\max}$.

```python
import numpy as np

A = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])  # rectangular, rank one
b = np.array([1.0, 2.0])
U, s, Vh = np.linalg.svd(A, full_matrices=False)
rcond = 1e-12
active = s > rcond * s[0]
reciprocal = np.divide(1.0, s, out=np.zeros_like(s), where=active)
x = Vh.T @ (reciprocal * (U.T @ b))

print(U.shape, s.shape, Vh.shape)
print(active.tolist())
print(np.round(x, 6).tolist())
print(np.allclose(A @ x, b))
np.testing.assert_allclose(x, np.linalg.pinv(A, rcond=rcond) @ b)
```

```text
(2, 2) (2,) (2, 3)
[True, False]
[0.071429, 0.142857, 0.214286]
True
```

Here $x=(1,2,3)/14$. The reduced shapes make the formula work for a rectangular
matrix; a square diagonal inverse with full-sized rectangular SVD factors can
have incompatible dimensions.

The cutoff is an assumption about which directions are resolvable. A larger
cutoff can discard a real but weak direction; a smaller cutoff can fit noise.
`lstsq` and `pinv` have their own cutoff defaults, so specify a policy when
numerical rank is important. For complex matrices, replace the transposes in
the derivation with conjugate transposes.

## A tiny residual can accompany a large solution error

For nonsingular $A$ and nonzero $b$, the relative forward error obeys

$$
\frac{\|\hat x-x\|_2}{\|x\|_2}
\leq
\kappa_2(A)\frac{\|b-A\hat x\|_2}{\|b\|_2}.
$$

A small residual helps only in combination with a moderate condition number.

```python
import numpy as np

A = np.diag([1.0, 1e-12])
b = np.array([1.0, 1e-12])
x = np.linalg.solve(A, b)
perturbed_b = np.array([1.0, 0.0])  # change one input by only 1e-12
x_hat = np.linalg.solve(A, perturbed_b)

relative_residual = np.linalg.norm(b - A @ x_hat) / np.linalg.norm(b)
relative_error = np.linalg.norm(x_hat - x) / np.linalg.norm(x)
print(x.tolist(), x_hat.tolist())
print(f"{relative_residual:.1e}")
print(f"{relative_error:.6f}")
print(f"{np.linalg.cond(A):.0e}")
```

```text
[1.0, 1.0] [1.0, 0.0]
1.0e-12
0.707107
1e+12
```

The data perturbation is tiny relative to $\|b\|_2$, yet the second solution
component changes by $1$. This is sensitivity of the problem, not evidence
that `solve` malfunctioned. Inspect conditioning and the accuracy of your input
data before interpreting many printed digits as reliable.

## Batched right-hand-side vectors need an explicit axis

NumPy 2.x treats `b` as a vector only when it is exactly one-dimensional;
otherwise it is interpreted as a stack of right-hand-side matrices.
For one vector per matrix, give the right-hand side a final singleton axis,
solve, and remove that axis.

```python
import numpy as np

A = np.array([[[2.0, 0.0], [0.0, 3.0]],
              [[4.0, 0.0], [0.0, 5.0]]])
b = np.array([[2.0, 6.0], [4.0, 10.0]])
x = np.linalg.solve(A, b[..., None])[..., 0]

print(A.shape, b.shape, x.shape)
print(x.tolist())
np.testing.assert_allclose((A @ x[..., None])[..., 0], b)
```

```text
(2, 2, 2) (2, 2) (2, 2)
[[1.0, 2.0], [1.0, 2.0]]
```

Without the extra axis, `b.shape == (2,2)` can be interpreted as two columns
shared across the batch, producing a different-shaped problem that may still
be accepted. Explicit axes make the intended pairing unambiguous.

See NumPy's [solve and its shape conventions](https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html),
[least squares and returned residuals](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html),
and [pseudoinverse](https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html) references.

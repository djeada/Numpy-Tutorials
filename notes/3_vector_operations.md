# Vectors: operations, geometry, and broadcasting

A real coordinate vector $v=(v_0,\ldots,v_{n-1})$ is an ordered list of $n$
components. A NumPy array of shape `(n,)` is a convenient representation,
but NumPy's shape rules are part of the calculation: they are not mathematical
row/column notation inferred from how an array is printed.

## Addition, scaling, inner products, and outer products

For real vectors $u,v\in\mathbb R^n$,

$$
(u+v)_i=u_i+v_i,\quad
(cu)_i=cu_i,\quad
(u\odot v)_i=u_iv_i,\quad
u^\top v=\sum_i u_iv_i.
$$

The Hadamard product $u\odot v$ is a vector. The dot product is a scalar.
The outer product $uv^\top$ is a matrix with entries $(uv^\top)_{ij}=u_iv_j$.

```python
import numpy as np

u = np.array([1, 2, 3])
v = np.array([4, 5, 6])
print((u + v).tolist())
print((2 * u).tolist())
print((u * v).tolist())
print(u @ v)
print(np.outer(u, v).tolist())
print((u[:, None] @ v[None, :]).shape)
```

```text
[5, 7, 9]
[2, 4, 6]
[4, 10, 18]
32
[[4, 5, 6], [8, 10, 12], [12, 15, 18]]
(3, 3)
```

The dot product is $1\cdot4+2\cdot5+3\cdot6=32$.
Scalar multiplication `2 * u` is different from an outer product; “outer”
does not mean multiplying a vector by a scalar.

## A one-dimensional transpose does not create a column

```python
import numpy as np

v = np.array([1, 2, 3])
row = v[None, :]
column = v[:, None]

print(v.shape, v.T.shape)
print(row.shape, column.shape)
print((v @ v).shape)
print((row @ column).tolist())
print((column @ row).tolist())
```

```text
(3,) (3,)
(1, 3) (3, 1)
()
[[14]]
[[1, 2, 3], [2, 4, 6], [3, 6, 9]]
```

Here `v @ v` is a scalar with shape `()`, whereas `row @ column` has shape
`(1,1)`. The values represent the same sum of squares, but their shapes affect
later operations.

## Length, projection, and angle connect the algebra to geometry

The three frequently used vector norms are

$$
\|v\|_1=\sum_i|v_i|,\qquad
\|v\|_2=\sqrt{\sum_i|v_i|^2},\qquad
\|v\|_\infty=\max_i|v_i|.
$$

For $v=(3,-4)$ these are $7$, $5$, and $4$.
A norm is nonnegative, vanishes only at zero, satisfies
$\|cv\|=|c|\|v\|$, and obeys $\|u+v\|\leq\|u\|+\|v\|$.
The formula $(\sum_i|v_i|^p)^{1/p}$ is a norm for $p\geq1$, not for $0<p<1$.

For nonzero $u$, the orthogonal projection of $v$ onto the line spanned by $u$ is

$$
p=\frac{u^\top v}{u^\top u}u,\qquad u^\top(v-p)=0.
$$

With $u=(1,2)$ and $v=(3,1)$, the coefficient is $(3+2)/(1+4)=1$,
so $p=(1,2)$ and the perpendicular residual is $(2,-1)$.

```python
import numpy as np

w = np.array([3.0, -4.0])
print([float(np.linalg.norm(w, ord=p)) for p in (1, 2, np.inf)])

u = np.array([1.0, 2.0])
v = np.array([3.0, 1.0])
projection = ((u @ v) / (u @ u)) * u
residual = v - projection
print(projection.tolist())
print(residual.tolist())
print(float(u @ residual))

a = np.array([1.0, 0.0])
b = np.array([1.0, 1.0])
cosine = (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
print(round(float(angle), 6))
```

```text
[7.0, 5.0, 4.0]
[1.0, 2.0]
[2.0, -1.0]
0.0
45.0
```

The angle formula is $\cos\theta=(a^\top b)/(\|a\|_2\|b\|_2)$.
Clipping protects `arccos` against a small rounding excursion outside $[-1,1]$.
It does not define the angle of a zero vector; check nonzero lengths first.

In three dimensions the cross product is another distinct operation:
$u\times v=(u_1v_2-u_2v_1,\ u_2v_0-u_0v_2,\ u_0v_1-u_1v_0)$.
Its length is the parallelogram area, and reversing its arguments reverses its sign.

```python
import numpy as np

u = np.array([1, 0, 0])
v = np.array([0, 1, 0])
print(np.cross(u, v).tolist())
print(np.cross(v, u).tolist())
print(u @ v)
```

```text
[0, 0, 1]
[0, 0, -1]
0
```

These perpendicular vectors have zero dot product but a nonzero cross product.

## Complex vectors need conjugation for a squared length

For $z\in\mathbb C^n$, the Hermitian inner product is
$\langle z,w\rangle=\sum_i\overline{z_i}w_i$.
Neither `@` nor `np.dot` conjugates a one-dimensional input.
`np.vdot` conjugates the first input and flattens both inputs.

```python
import numpy as np

z = np.array([1 + 1j, 1 - 1j])
print(complex(z @ z))
print(complex(np.vdot(z, z)))
print(float(np.linalg.norm(z)))
```

```text
0j
(4+0j)
2.0
```

The unconjugated squares cancel, but
$|1+i|^2+|1-i|^2=2+2=4$. Thus `z @ z` is not a complex vector's squared
Euclidean length. Because `vdot` flattens, it also does not perform a batch of
separate inner products on a matrix.

## Broadcasting follows trailing axes, not row labels

NumPy aligns shapes from the right. At each aligned axis the lengths must match
or one must be $1$; missing leading axes act as length $1$.
For example, `(3,1)` and `(4,)` produce `(3,4)`, while `(3,)` and `(4,)`
cannot be added.

A dangerous case is a square matrix: a scale vector can fit either interpretation,
so a mistaken axis produces an answer without an exception.

```python
import numpy as np

X = np.array([[1, 2], [3, 4]])  # rows: observations; columns: features
scale = np.array([10, 100])

by_feature = X * scale          # (2, 2) times (2,): scale columns
by_observation = X * scale[:, None]  # (2, 2) times (2, 1): scale rows
print(by_feature.tolist())
print(by_observation.tolist())

u = np.array([1, 2, 3])
print((u + u[:, None]).tolist())
```

```text
[[10, 200], [30, 400]]
[[10, 20], [300, 400]]
[[2, 3, 4], [3, 4, 5], [4, 5, 6]]
```

The last expression forms every pairwise sum, $S_{ij}=u_i+u_j$.
Its shape is `(3,3)`, not `(3,)`. Write down what each axis means before relying
on shapes that happen to fit.

## Normalizing a batch requires a policy for zero vectors

For rows $x_i$, normalization means $y_i=x_i/\|x_i\|_2$.
A zero row has no unit direction. Here we explicitly choose to leave it zero.

```python
import numpy as np

X = np.array([[3.0, 4.0], [0.0, 0.0], [0.0, 5.0]])
lengths = np.linalg.norm(X, axis=1, keepdims=True)  # shape (3, 1)
unit = np.divide(X, lengths, out=np.zeros_like(X), where=lengths > 0)

print(lengths.ravel().tolist())
print(unit.tolist())
print(np.linalg.norm(unit, axis=1).tolist())
```

```text
[5.0, 0.0, 5.0]
[[0.6, 0.8], [0.0, 0.0], [0.0, 1.0]]
[1.0, 0.0, 1.0]
```

`keepdims=True` retains the reduced axis at length one so each length broadcasts
across its own row. For this `(3,2)` input, dividing by lengths of shape `(3,)`
would fail. For a square input, an incorrect division could silently succeed.

## Pairwise distances create a much larger intermediate

For observations $x_i\in\mathbb R^d$,

$$
D_{ij}=\|x_i-x_j\|_2
=\sqrt{\sum_{k=0}^{d-1}(X_{ik}-X_{jk})^2}.
$$

```python
import numpy as np

X = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
difference = X[:, None, :] - X[None, :, :]  # (3, 1, 2) - (1, 3, 2)
distance = np.linalg.norm(difference, axis=-1)

print(difference.shape, distance.shape)
print(distance.tolist())
np.testing.assert_allclose(distance, distance.T)
np.testing.assert_allclose(np.diag(distance), 0)
```

```text
(3, 3, 2) (3, 3)
[[0.0, 3.0, 4.0], [3.0, 0.0, 5.0], [4.0, 5.0, 0.0]]
```

The off-diagonal $5$ is $\sqrt{3^2+4^2}$. Broadcasting avoids copying the inputs
to tiled arrays, but the subtraction still allocates `difference`.
At $n=10{,}000$, $d=128$, that `float64` intermediate alone requires
$n^2d\cdot8=102.4$ GB in decimal units. Compute in blocks when this is too large;
a vectorized expression is not automatically a memory-efficient algorithm.

See NumPy's [broadcasting guide](https://numpy.org/doc/stable/user/basics.broadcasting.html),
[vector and matrix norms](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html),
and [complex inner products](https://numpy.org/doc/stable/reference/generated/numpy.vdot.html).

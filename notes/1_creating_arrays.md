# Creating arrays: shape, values, and arithmetic

An array has three separate properties: its **shape** says how values are indexed,
its **dtype** says how each value is represented, and its **data** contains the
values. Choosing the wrong dtype can change an answer even when the shape is right.

For a shape $(n_0,\ldots,n_{d-1})$, the number of elements is
$N=\prod_{k=0}^{d-1}n_k$. The number of axes, `ndim`, is $d$.
This is different from matrix rank, which counts independent rows or columns.

## A scalar, a vector, and a column matrix have different shapes

The mathematical vector $(2,4,6)$ can be stored as `(3,)`, `(1,3)`, or `(3,1)`.
These representations participate differently in broadcasting and matrix products.
A one-dimensional array has no row or column orientation.

```python
import numpy as np

scalar = np.array(7)
vector = np.array([2, 4, 6])
row = np.array([[2, 4, 6]])
column = np.array([[2], [4], [6]])

for a in (scalar, vector, row, column):
    print(a.shape, a.ndim, a.size)
print(vector.T.shape)
```

```text
() 0 1
(3,) 1 3
(1, 3) 2 3
(3, 1) 2 3
(3,)
```

The transpose of a one-dimensional array still has shape `(3,)`.
Use `vector[:, None]` to introduce a column axis. Nested lists must be rectangular
to describe an ordinary numeric matrix.

## Constructors encode different mathematical objects

`np.zeros`, `np.ones`, and `np.full` fill every entry. `np.eye` places ones on a
diagonal. An all-ones matrix is an identity for **elementwise multiplication**;
an identity matrix is an identity for **matrix multiplication**.

```python
import numpy as np

a = np.array((2, 4, 6), dtype=np.float64)
print(np.zeros((2, 3), dtype=np.int64).tolist())
print(np.ones((2, 3), dtype=np.int64).tolist())
print(np.full((2, 3), 0.5).tolist())
print((np.eye(3) @ a).tolist())
print((np.ones((3, 3)) @ a).tolist())
```

```text
[[0, 0, 0], [0, 0, 0]]
[[1, 1, 1], [1, 1, 1]]
[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
[2.0, 4.0, 6.0]
[12.0, 12.0, 12.0]
```

In the last product every row computes $2+4+6=12$. Filling a matrix with ones
does not make it the identity transformation.

## Sampling a function: count and spacing are different choices

For $n\geq2$ samples including both endpoints of $[a,b]$,

$$
t_k=a+k\frac{b-a}{n-1},\qquad k=0,\ldots,n-1.
$$

Thus five samples on $[0,1]$ have spacing $1/4$, not $1/5$.
For a periodic interval, `endpoint=False` avoids duplicating the start at the end.

```python
import numpy as np

t, dt = np.linspace(0.0, 1.0, 5, retstep=True)
periodic_t = np.linspace(0.0, 1.0, 4, endpoint=False)
print(t.tolist())
print(float(dt))
print((t**2).tolist())  # f(t) = t^2 evaluated at every grid point
print(periodic_t.tolist())
print(np.arange(0, 10, 2).tolist())
```

```text
[0.0, 0.25, 0.5, 0.75, 1.0]
0.25
[0.0, 0.0625, 0.25, 0.5625, 1.0]
[0.0, 0.25, 0.5, 0.75]
[0, 2, 4, 6, 8]
```

Use integer `arange` for index sequences; use `linspace` when the sample count
and endpoint policy are the specification. Decimal steps are generally not exact
binary fractions. Even a grid that looks right when printed can fail exact equality:

```python
import numpy as np

t = np.arange(0.0, 1.0, 0.1)
print(repr(float(t[3])))
print(bool(t[3] == 0.3))
print(bool(np.isclose(t[3], 0.3, rtol=0, atol=1e-15)))
```

```text
0.30000000000000004
False
True
```

`linspace` controls the count; it does not make floating-point arithmetic exact.
Choose comparison tolerances from the scale and accuracy of the calculation.

## Fixed-width integers can overflow before you convert the result

An `int8` stores integers from $-128$ to $127$. Array arithmetic in that dtype
wraps modulo $256$: the mathematical result $130$ becomes $-126$.
Python's arbitrary-precision integers do not impose this limit.

```python
import numpy as np

a = np.array([120, 127], dtype=np.int8)
increment = np.array([10, 1], dtype=np.int8)
wrapped = a + increment
wide = a.astype(np.int64) + increment
too_late = wrapped.astype(np.int64)

print(wrapped.tolist())
print(wide.tolist())
print(too_late.tolist())
```

```text
[-126, -128]
[130, 128]
[-126, -128]
```

**Convert before the operation.** Widening an already-overflowed answer cannot
recover the original values. `int64` also has a finite range; it merely moves the
limit. Likewise, `float32` saves memory but stores less precision than `float64`.

## Assignment and “like” constructors keep the destination dtype

An array does not automatically become floating point when a fractional value is
assigned to it. A mixed list such as `[1, 2.5]` would infer a floating dtype when
creating a new array; assigning into an existing integer array is different.

```python
import numpy as np

a = np.array([1, 2, 3], dtype=np.int64)
a[0] = 1.9
print(a.tolist())
print((a / 2).tolist())
print(np.full_like(a, 0.5).tolist())
print(np.full_like(a, 0.5, dtype=np.float64).tolist())
```

```text
[1, 2, 3]
[0.5, 1.0, 1.5]
[0, 0, 0]
[0.5, 0.5, 0.5]
```

Here assignment truncates $1.9$ toward zero. `full_like` inherits `a.dtype` unless
overridden. Allocate a floating array at the start when fractions are part of the
problem. Integer in-place operations such as `a += 0.5` can instead reject the
cast; do not assume all assignment-like syntax has identical casting rules.

## Copies, aliases, and uninitialized storage

For an ordinary `ndarray` with no dtype or order conversion requested,
`np.asarray(a)` can return `a` itself. `np.array(a)` copies by default.
Passing an array through a function does not isolate its contents.

```python
import numpy as np

a = np.array([1.0, 2.0])
alias = np.asarray(a)
independent = np.array(a)
alias[0] = 99

print(alias is a)
print(a.tolist())
print(independent.tolist())

# Every entry is written before the buffer is read.
table = np.empty((3, 2), dtype=np.float64)
table[:, 0] = [0, 1, 2]
table[:, 1] = table[:, 0] ** 2
print(table.tolist())
print(table.nbytes)
```

```text
True
[99.0, 2.0]
[1.0, 2.0]
[[0.0, 0.0], [1.0, 1.0], [2.0, 4.0]]
48
```

`empty` allocates storage without initializing the numeric entries. Reading them
before writing them is a bug. This table uses $3\cdot2\cdot8=48$ bytes for its
element buffer; `nbytes` excludes the Python object's overhead.

For random input, create a local generator, for example
`rng = np.random.default_rng(42)` and then `rng.normal(size=(3, 2))`.
A seed selects a reproducible stream in a given environment, not a particular
sample mean or variance.

The NumPy references explain [array creation](https://numpy.org/doc/stable/user/basics.creation.html)
and [numeric types and overflow](https://numpy.org/doc/stable/user/basics.types.html).

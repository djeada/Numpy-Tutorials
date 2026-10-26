# Combining and splitting arrays: preserve the meaning of each axis

Joining arrays is a modeling choice. Does the new data add observations, add
features to existing observations, or add a new experiment axis?
Those choices call for different shapes, even when all inputs contain the same
number of elements.

## Concatenate extends an axis; stack introduces an axis

If $A,B\in\mathbb R^{m\times n}$, then

$$
\operatorname{concatenate}_0(A,B)\in\mathbb R^{2m\times n},\qquad
\operatorname{concatenate}_1(A,B)\in\mathbb R^{m\times2n}.
$$

By contrast, stacking two such matrices along axis zero produces an array
$S\in\mathbb R^{2\times m\times n}$ with
$S_{0,i,j}=A_{i,j}$ and $S_{1,i,j}=B_{i,j}$.

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

for result in (np.concatenate((A, B), axis=0),
               np.concatenate((A, B), axis=1),
               np.stack((A, B), axis=0),
               np.stack((A, B), axis=-1)):
    print(result.shape, result.tolist())
```

```text
(4, 2) [[1, 2], [3, 4], [5, 6], [7, 8]]
(2, 4) [[1, 2, 5, 6], [3, 4, 7, 8]]
(2, 2, 2) [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
(2, 2, 2) [[[1, 5], [2, 6]], [[3, 7], [4, 8]]]
```

The last two shapes happen to be equal because every dimension is two.
Their axis meanings and values differ: the input selector is first in one and
last in the other.

| Intended meaning, with observations in rows | Operation | Required agreement |
| --- | --- | --- |
| Add observations | `concatenate(..., axis=0)` | Same feature count and feature definitions |
| Add features | `concatenate(..., axis=1)` | Same observations in the same row order |
| Keep experiments separate | `stack(..., axis=0)` | Exactly matching input shapes |

Concatenation requires equality on every axis except the joined axis.
Stack requires equal input shapes. Neither operation broadcasts mismatched
inputs to make them fit. Matching lengths also do not verify that row identities
or feature units agree.

## The stacking shortcuts treat one-dimensional inputs differently

`hstack` concatenates one-dimensional inputs on axis zero, but uses axis one
for inputs with at least two axes. `vstack` first promotes a one-dimensional
input to a row. `column_stack` promotes it to a column.

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(np.hstack((a, b)).tolist())
print(np.vstack((a, b)).tolist())
print(np.column_stack((a, b)).tolist())
print(np.dstack((a, b)).shape)
print(np.dstack((a, b)).tolist())
```

```text
[1, 2, 3, 4, 5, 6]
[[1, 2, 3], [4, 5, 6]]
[[1, 4], [2, 5], [3, 6]]
(1, 3, 2)
[[[1, 4], [2, 5], [3, 6]]]
```

`dstack` promotes one-dimensional `(n,)` inputs to `(1,n,1)` and
two-dimensional `(m,n)` inputs to `(m,n,1)`, then concatenates on axis two.
It does not universally mean “add a new axis.”

```python
import numpy as np

A = np.arange(4).reshape(2, 2, 1)
B = A + 10
print(np.dstack((A, B)).shape)
print(np.stack((A, B), axis=-1).shape)
```

```text
(2, 2, 2)
(2, 2, 1, 2)
```

For already three-dimensional inputs, `dstack` extends the existing third axis.
An explicit `stack` or `concatenate` often makes axis intent easier to read.

## Joining copies values; splitting can expose views into the result

Ordinary `stack` and `concatenate` calls allocate a destination and copy input
values into it. Contiguous inputs do not make the joined result share their
storage. An explicit `out` buffer can supply the destination, but values still
have to be placed into that buffer.

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
joined = np.concatenate((a, b))
stacked = np.stack((a, b))
parts = np.split(joined, 2)

print(np.shares_memory(a, joined), np.shares_memory(a, stacked))
print(np.shares_memory(joined, parts[0]))
parts[0][0] = -10
print(joined.tolist())
print(a.tolist())
```

```text
False False
True
[-10, 2, 3, 4, 5, 6]
[1, 2, 3]
```

Splitting the joined buffer produces views here. Mutating a split changes
`joined`, but cannot cross the earlier copy boundary back into `a`.
Copy a split explicitly when it needs independent storage.

## Append returns a new array and flattens only when axis is omitted

Unlike a Python list's `append` method, `np.append` does not grow its input
in place. With the default `axis=None` it flattens the inputs first.
With an explicit axis it follows concatenation shape rules.

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6]])
print(np.append(A, B).tolist())
print(np.append(A, B, axis=0).tolist())
print(A.tolist())

# An untyped empty list becomes a float array and promotes the result.
promoted = np.append(np.array([1, 2], dtype=np.int64), [])
print(promoted.dtype.name, promoted.tolist())
```

```text
[1, 2, 3, 4, 5, 6]
[[1, 2], [3, 4], [5, 6]]
[[1, 2], [3, 4]]
float64 [1.0, 2.0]
```

Do not discard the return value if you intend to use the appended result.
Also consider dtype promotion: an empty append can change dtype even though
it adds no values.

## Repeated appends make a linear data-collection task quadratic

Appending one element at a time creates results of lengths $1,2,\ldots,N$.
The total number of values written across these allocations is

$$
1+2+\cdots+N=\frac{N(N+1)}2,
$$

rather than just $N$. Collect chunks and concatenate once, or preallocate when
the final size is known.

```python
import numpy as np

chunks = [np.array([[1, 2], [3, 4]]), np.array([[5, 6]])]
combined = np.concatenate(chunks, axis=0)

row_count = sum(chunk.shape[0] for chunk in chunks)
preallocated = np.empty((row_count, 2), dtype=np.int64)
start = 0
for chunk in chunks:
    stop = start + len(chunk)
    preallocated[start:stop] = chunk
    start = stop

print(combined.tolist())
print(np.array_equal(combined, preallocated))
```

```text
[[1, 2], [3, 4], [5, 6]]
True
```

The preallocated buffer is read only after every row has been filled.
For an empty chunk collection, define the result explicitly, for example
`np.empty((0, 2), dtype=np.int64)`; concatenating no arrays raises an error.
When dtypes differ, choose a destination dtype that represents every chunk
without unintended truncation.

## Equal sections and specified cut positions are different contracts

`split(a, 3)` requires three equal-length sections along the chosen axis.
`array_split(a, 3)` allows unequal sections, putting the larger sections first.
A list such as `[3,7]` specifies cut positions instead of a section count.

```python
import numpy as np

a = np.arange(10)
try:
    np.split(a, 3)
except ValueError:
    print("10 elements cannot form 3 equal sections")

unequal = np.array_split(a, 3)
cuts = np.split(a, [3, 7])
print([part.tolist() for part in unequal])
print([part.tolist() for part in cuts])
print([bool(np.shares_memory(a, part)) for part in unequal])
```

```text
10 elements cannot form 3 equal sections
[[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]
[[0, 1, 2], [3, 4, 5, 6], [7, 8, 9]]
[True, True, True]
```

For two-dimensional inputs, `hsplit` splits columns and `vsplit` splits rows.
Using `split(..., axis=...)` makes the same choice explicit for general arrays.
Cuts beyond the end or repeated cut positions can produce empty sections.

## Channel stacking and block-diagonal assembly are different mathematics

Depth stacking places arrays along a channel axis.
A block-diagonal matrix instead represents separate linear transformations:

$$
D=\begin{bmatrix}A&0\\0&B\end{bmatrix},\qquad
D\begin{bmatrix}x\\y\end{bmatrix}
=\begin{bmatrix}Ax\\By\end{bmatrix}.
$$

Build the zero blocks explicitly with `np.block`.

```python
import numpy as np

A = np.array([[2.0, 1.0], [1.0, 3.0]])
B = np.array([[4.0]])
D = np.block([[A, np.zeros((2, 1))],
              [np.zeros((1, 2)), B]])
v = np.array([1.0, 2.0, 3.0])

print(D.tolist())
print((D @ v).tolist())
np.testing.assert_allclose(D @ v, np.concatenate((A @ v[:2], B @ v[2:])))
print(np.dstack((A, A)).shape)
```

```text
[[2.0, 1.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 4.0]]
[4.0, 7.0, 12.0]
(2, 2, 2)
```

The block matrix has shape `(3,3)` and no interactions between its two groups
of variables. The depth stack has three axes and represents two channels of
a `(2,2)` array; it is not a block-diagonal matrix.

## Shuffle paired data with one shared ordering

When rows of `X` correspond to entries of `y`, separately shuffling the two
arrays destroys their correspondence. Generate one permutation and apply it
to both before taking matching splits.

```python
import numpy as np

X = np.arange(12).reshape(6, 2)
y = 100 + np.arange(6)
rng = np.random.default_rng(42)
order = rng.permutation(len(X))
X_shuffled, y_shuffled = X[order], y[order]

X_train, X_test = X_shuffled[:4], X_shuffled[4:]
y_train, y_test = y_shuffled[:4], y_shuffled[4:]
print(np.array_equal(y_shuffled, 100 + X_shuffled[:, 0] // 2))
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
```

```text
True
(4, 2) (4,) (2, 2) (2,)
```

The first check verifies row identity after shuffling. A matching shape would
not detect incorrectly paired labels.

See NumPy's [stack](https://numpy.org/doc/stable/reference/generated/numpy.stack.html),
[depth stack](https://numpy.org/doc/stable/reference/generated/numpy.dstack.html), and
[array splitting](https://numpy.org/doc/stable/reference/generated/numpy.array_split.html) references.

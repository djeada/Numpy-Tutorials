# Accessing and modifying elements: selection versus shared storage

Indexing answers two questions: **which values are selected**, and **whether the
selection shares storage with the original**. The second question determines
whether a later assignment changes your input.

All indices below start at zero. For a matrix $A\in\mathbb R^{m\times n}$,
`A[i, j]` denotes $A_{ij}$, with $0\leq i<m$ and $0\leq j<n$.

## A slice excludes its stop; an integer index removes an axis

For a positive step, `a[start:stop:step]` visits indices
$start,start+step,\ldots$ strictly below `stop`. With a negative step the direction
and default endpoints reverse. `a[::-1]` therefore reverses an axis.

```python
import numpy as np

a = np.arange(8)
print(a[1:7:2].tolist())
print(a[-1])
print(a[::-1].tolist())
print(a[5:1:-2].tolist())

A = np.arange(12).reshape(3, 4)
print(A[1, 2])
print(A[:2, 1:3].tolist())
print(A[1].shape, A[1:2].shape)
print(A[:, 1].shape, A[:, 1:2].shape)
```

```text
[1, 3, 5]
7
[7, 6, 5, 4, 3, 2, 1, 0]
[5, 3]
6
[[1, 2], [5, 6]]
(4,) (1, 4)
(3,) (3, 1)
```

`A[:, 1:2]` retains a column axis. This matters when the result will be
broadcast against another matrix. A slice extending beyond the end is clipped;
a single out-of-bounds integer index raises `IndexError`.

## Basic slices are views; integer-array selections are copies

For ordinary numeric arrays, a basic slice can be described by an offset and
strides into the existing buffer. Selecting arbitrary integer positions gathers
values into a separate buffer.

```python
import numpy as np

a = np.array([10, 20, 30, 40])
window = a[1:3]       # basic slice: a view
gathered = a[[1, 2]]  # advanced indexing: a copy
detached = a[1:3].copy()

window[0] = -1
gathered[1] = 999
print(a.tolist())
print(gathered.tolist())
print(detached.tolist())
print(np.shares_memory(a, window), np.shares_memory(a, gathered))
```

```text
[10, -1, 30, 40]
[20, 999]
[20, 30]
True False
```

The assignment `window[0] = -1` updates `a[1]`. Modifying `gathered` or
`detached` does not update `a`. Use `np.shares_memory` when investigating a
specific pair of arrays; `b.base is a` is not a reliable general test because
a view may refer to an earlier owner in a chain of views.

These copying rules describe **reading** a selection. Direct indexed assignment
has different behavior: `a[[1, 2]] = values` writes to `a`.

## Chained indexing can silently throw away an update

Boolean indexing also gathers a copy. Consequently, `a[mask][0] = value` updates
a temporary, while `a[mask] = value` updates the original array.

```python
import numpy as np

a = np.array([1, 6, 8, 2])
mask = a > 5
a[mask][0] = 100
print(a.tolist())

a[mask] = -1
print(a.tolist())

# To change only the first matching entry, select its original index.
b = np.array([1, 6, 8, 2])
first = np.flatnonzero(b > 5)[0]  # assumes at least one match
b[first] = 100
print(b.tolist())
```

```text
[1, 6, 8, 2]
[1, -1, -1, 2]
[1, 100, 8, 2]
```

If there might be no match, check the size of `np.flatnonzero(...)` before taking
element zero. Also remember that `mask` records the condition when it was
computed; changing `a` does not recompute an existing mask.

## Two index arrays select pairs, not a rectangular submatrix

Given row indices $r_k$ and column indices $c_k$, `A[rows, cols]` selects
$A_{r_k,c_k}$. To select every combination $A_{r_i,c_j}$, use `np.ix_`
or reshape the row indices to a column so that the indices broadcast.

```python
import numpy as np

A = np.arange(12).reshape(3, 4)
rows = np.array([0, 2])
cols = np.array([1, 3])

paired = A[rows, cols]
rectangle = A[np.ix_(rows, cols)]
print(A.tolist())
print(paired.tolist(), paired.shape)
print(rectangle.tolist(), rectangle.shape)
np.testing.assert_array_equal(rectangle, A[rows[:, None], cols])
```

```text
[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
[1, 11] (2,)
[[1, 3], [9, 11]] (2, 2)
```

The first selection is $(A_{0,1},A_{2,3})=(1,11)$.
The rectangle also includes $A_{0,3}=3$ and $A_{2,1}=9$.
Both selections above are advanced indexing and therefore return copies.

## Repeated indices do not make `+=` a counting operation

An expression like `a[idx] += 1` gathers selected values, modifies the temporary,
and writes values back. When an index repeats, this does not accumulate one
increment per occurrence. `np.add.at` performs the repeated updates.

```python
import numpy as np

buffered = np.zeros(3, dtype=np.int64)
buffered[[0, 0, 2]] += 1
print(buffered.tolist())

accumulated = np.zeros(3, dtype=np.int64)
np.add.at(accumulated, [0, 0, 2], 1)
print(accumulated.tolist())
```

```text
[1, 0, 1]
[2, 0, 1]
```

This distinction matters in histograms, graph degrees, and scatter-add algorithms.
Do not rely on a particular write order when repeated indices are assigned
different values.

A concrete aggregation problem: events have group labels $g_i$ and amounts $w_i$.
The desired total for group $j$ is

$$
t_j=\sum_{\{i:g_i=j\}}w_i.
$$

```python
import numpy as np

group = np.array([0, 2, 0, 2])
amount = np.array([5, 7, 11, 13])
totals = np.zeros(3, dtype=np.int64)
np.add.at(totals, group, amount)

print(totals.tolist())
print(np.bincount(group, weights=amount, minlength=3).tolist())
```

```text
[16, 0, 20]
[16.0, 0.0, 20.0]
```

Group zero receives $5+11=16$, group two receives $7+13=20$, and group one
receives zero. `bincount` requires nonnegative integer labels; its weighted
form returns floating values, so consider precision if accumulating very large
integers.

## Broadcasting assignments still obey selection shape

Assignment does not require a separate right-hand value for every selected entry.
A compatible shape can broadcast: a vector of column values can fill several rows.

```python
import numpy as np

A = np.zeros((3, 4), dtype=np.int64)
A[:2, 1:3] = [10, 20]  # selection shape (2, 2), values shape (2,)
print(A.tolist())

A[:, 0] = [1, 2, 3]
print(A.tolist())
```

```text
[[0, 10, 20, 0], [0, 10, 20, 0], [0, 0, 0, 0]]
[[1, 10, 20, 0], [2, 10, 20, 0], [3, 0, 0, 0]]
```

The shape and dtype of the destination remain fixed. Assigning a fractional value
to this integer array would not turn it into a floating array.

See NumPy's [indexing rules](https://numpy.org/doc/stable/user/basics.indexing.html),
[copies and views](https://numpy.org/doc/stable/user/basics.copies.html), and
[unbuffered `ufunc.at` updates](https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html).

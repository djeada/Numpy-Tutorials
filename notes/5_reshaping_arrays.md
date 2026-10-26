# Reshaping arrays: index order, axis meaning, and memory

Reshaping preserves the number of elements:

$$
\prod_i n_i=\prod_j m_j.
$$

That condition makes a shape possible; it does not make the new axes meaningful.
Changing `(height, width, channels)` into `(channels, height, width)` requires
an axis permutation, not merely a new tuple with the same product.

## Reshape follows an enumeration of the values

In C order, the last index varies fastest. For a two-dimensional target with
$n$ columns, flat position $k$ becomes

$$
i=\lfloor k/n\rfloor,\qquad j=k\bmod n.
$$

In Fortran order, the first index varies fastest. `order="C"` and `order="F"`
describe how indices are traversed when reading and placing values; they do not
by themselves promise a particular physical memory layout for every result.

```python
import numpy as np

a = np.arange(6)
print(a.reshape(2, 3, order="C").tolist())
print(a.reshape(2, 3, order="F").tolist())
print(a.reshape(3, -1).tolist())
print(a.reshape(1, -1).shape, a.reshape(-1, 1).shape)
```

```text
[[0, 1, 2], [3, 4, 5]]
[[0, 2, 4], [1, 3, 5]]
[[0, 1], [2, 3], [4, 5]]
(1, 6) (6, 1)
```

`-1` asks NumPy to infer one dimension from the element count. It can appear at
most once. Six elements can form `(3,2)`, but cannot form `(4,k)` for any
integer $k$.

## Transpose changes which index names a value

For a matrix, transpose means $(A^\top)_{ij}=A_{ji}$.
Reshaping in C order instead preserves the C-order flat sequence.

```python
import numpy as np

A = np.arange(6).reshape(2, 3)
transposed = A.T
reshaped = A.reshape(3, 2)

print(A.tolist())
print(transposed.tolist())
print(reshaped.tolist())
print(np.array_equal(transposed, reshaped))
print(np.shares_memory(A, transposed), np.shares_memory(A, reshaped))
```

```text
[[0, 1, 2], [3, 4, 5]]
[[0, 3], [1, 4], [2, 5]]
[[0, 1], [2, 3], [4, 5]]
False
True True
```

Both results have shape `(3,2)` and share storage with `A`, yet contain
different arrangements. Shape and memory sharing do not establish semantic
equivalence.

## Image channels: a plausible shape can hide scrambled data

Suppose an image batch has shape `(batch, height, width, channels)`.
To make it `(batch, channels, height, width)`, the desired identity is

$$
Y_{b,c,h,w}=X_{b,h,w,c}.
$$

Use `moveaxis` or `transpose` to express that identity.

```python
import numpy as np

image = np.array([[
    [[10, 11, 12], [20, 21, 22]],
    [[30, 31, 32], [40, 41, 42]],
]])  # shape (1, 2, 2, 3); each pixel lists its three channels

correct = np.moveaxis(image, -1, 1)
scrambled = image.reshape(1, 3, 2, 2)

print(image.shape, correct.shape, scrambled.shape)
print(correct[0, 0].tolist())    # channel zero over all pixels
print(scrambled[0, 0].tolist())
print(np.array_equal(correct, image.transpose(0, 3, 1, 2)))
print(np.shares_memory(image, correct))
```

```text
(1, 2, 2, 3) (1, 3, 2, 2) (1, 3, 2, 2)
[[10, 20], [30, 40]]
[[10, 11], [12, 20]]
True
True
```

The wrong reshape mixes different channels and pixels. A downstream model might
accept the shape and still receive incorrect data. If a consumer requires a
C-contiguous channel-first buffer, use `np.ascontiguousarray(correct)` after
moving the axis; that may require a copy.

## Strides explain how a view can avoid moving elements

For a numeric array with byte strides $s_0,\ldots,s_{d-1}$, the byte address at
index $(i_0,\ldots,i_{d-1})$ is

$$
\text{address}(i)=\text{address of first logical element}+\sum_k i_ks_k.
$$

Strides can be negative or skip elements. Arrays are not all contiguous grids
in memory.

```python
import numpy as np

A = np.arange(12, dtype=np.int64).reshape(3, 4)
print(A.itemsize, A.strides)
print(A.T.strides)
print(A[:, ::-1].strides)
print(A[:, ::-1][0].tolist())

# (row=2, col=1) lies 2*32 + 1*8 = 72 bytes from A[0, 0].
print(2 * A.strides[0] + A.strides[1], A[2, 1])
```

```text
8 (32, 8)
(8, 32)
(32, -8)
[3, 2, 1, 0]
72 9
```

Transposition swaps strides. Reversing columns starts at the last column and
walks backward by eight bytes. Creating these views changes metadata; a later
operation that computes or copies values still has to visit those values.

## Reshape can copy, and noncontiguous does not always mean “must copy”

`reshape` returns a view when the requested layout can be expressed with strides;
otherwise it copies. `ravel` produces a contiguous one-dimensional result,
copying when necessary. `flatten` always copies.

A noncontiguous array can still reshape without a copy when its stride pattern
is regular enough:

```python
import numpy as np

A = np.arange(24).reshape(4, 6)
stepped = A[:, ::2]  # keep columns 0, 2, 4
flat_view = stepped.reshape(-1)
flat_contiguous = stepped.ravel()
flat_copy = stepped.flatten()

print(stepped.flags.c_contiguous)
print(np.shares_memory(stepped, flat_view))
print(np.shares_memory(stepped, flat_contiguous))
print(np.shares_memory(stepped, flat_copy))
flat_view[0] = -99
print(A[0, 0], flat_contiguous[0], flat_copy[0])
```

```text
False
True
False
False
-99 0 0
```

The selected values are every second element of the original buffer, including
across row boundaries. A single stride can describe the flattened view.
By contrast, flattening a transposed rectangular matrix in C order generally
needs a different traversal:

```python
import numpy as np

A = np.arange(12).reshape(3, 4)
T = A.T
flat = T.reshape(-1)  # default C-order enumeration of T

print(flat.tolist())
print(np.shares_memory(T, flat))
```

```text
[0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
False
```

Check sharing with `np.shares_memory` if mutation or memory use depends on it.
A non-`None` `base` can refer to an intermediate copy, so it does not prove that
a result shares memory with the particular input you care about.

## Singleton axes and empty arrays need explicit intent

Adding an axis of length one does not add elements. Removing every singleton
axis can accidentally remove a batch axis when the batch happens to contain
one item. Prefer `squeeze(axis=...)` when a particular axis is meant to go away.

```python
import numpy as np

X = np.zeros((1, 2, 1))
print(X.squeeze().shape)
print(X.squeeze(axis=2).shape)
print(np.expand_dims(np.array([5, 6]), axis=0).shape)

for a, target in ((np.arange(6), (4, -1)),
                  (np.empty((0, 3)), (0, -1))):
    try:
        a.reshape(target)
    except ValueError:
        print("cannot infer", target)
print(np.empty((0, 3)).reshape(0, 1, 3).shape)
```

```text
(2,)
(1, 2)
(1, 2)
cannot infer (4, -1)
cannot infer (0, -1)
(0, 1, 3)
```

The empty case is ambiguous: $0\cdot k=0$ for every $k$, so NumPy cannot infer
a unique missing dimension. Explicit zero-size shapes are valid.

## Resize is not reshape with padding

`np.resize` can change the number of elements. When enlarging a nonempty array,
it repeats its flattened data; it does not fill the new entries with zeros.
This is also different from the mutating `ndarray.resize` method.

```python
import numpy as np

a = np.array([1, 2, 3])
print(np.resize(a, (2, 4)).tolist())
print(np.pad(a, (0, 5), constant_values=0).reshape(2, 4).tolist())
print(a.tolist())
```

```text
[[1, 2, 3, 1], [2, 3, 1, 2]]
[[1, 2, 3, 0], [0, 0, 0, 0]]
[1, 2, 3]
```

For padding, state the fill policy with `np.pad` or allocate and fill a destination.
For rearranging axes, use an axis permutation. For regrouping a fixed sequence
of elements, use `reshape`.

See NumPy's [reshape](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html),
[moveaxis](https://numpy.org/doc/stable/reference/generated/numpy.moveaxis.html), and
[copies and views](https://numpy.org/doc/stable/user/basics.copies.html) references.

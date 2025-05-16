## Joining and Splitting Arrays

In NumPy, manipulating the structure of arrays is a common operation. Whether combining multiple arrays into one or splitting a single array into several parts, NumPy provides a set of intuitive functions to achieve these tasks efficiently. Understanding how to join and split arrays is essential for organizing data, preparing it for analysis, and optimizing computational performance. This guide covers various methods to join and split arrays, offering detailed explanations and practical examples to help you utilize these tools effectively.

### Understanding Axes, Dimensions & Matrices
Mathematicians talk about *entries* of a matrix $A\in \mathbb R^{m\times n}$ using two indices: rows $i$ and columns $j$.  NumPy generalises this idea to an arbitrary‑rank *tensor* whose **shape** is a tuple `(d₀, d₁, …, dₖ₋₁)`.  Each position in that tuple is called an **axis**:

| Rank  | Typical maths object | Shape example | Axis 0 meaning | Axis 1 meaning | Axis 2 meaning |
| ----- | -------------------- | ------------- | -------------- | -------------- | -------------- |
|  0‑D  | scalar               | `()`          | –              | –              | –              |
|  1‑D  | vector $v\_i$        | `(n,)`        | elements       | –              | –              |
|  2‑D  | matrix $A_{ij}$      | `(m, n)`      | **rows** $i$   | **cols** $j$   | –              |
|  3‑D  | stack of matrices    | `(k, m, n)`   | matrix index   | rows           | cols           |

**Axis conventions**

* `axis = 0` → *first* index ("down" in a 2‑D print‑out).
* `axis = 1` → *second* index ("across").
* Higher axes follow lexicographic order.

Thus, with two $m\times n$ matrices $A, B$:

* **`np.stack((A, B), axis=0)`** creates a tensor $T\in\mathbb R^{2\times m\times n}$.  Here `T[0] == A`, `T[1] == B`.
* **`np.stack((A, B), axis=1)`** yields $U\in\mathbb R^{m\times 2\times n}$ where `U[i,0] == A[i]`, `U[i,1] == B[i]`.

Likewise, reduction operations interpret `axis` in the same way: e.g. `A.sum(axis=0)` collapses rows and returns the column sums (a length‑`n` vector).

### Stacking Arrays

Stacking is the technique of **joining a sequence of arrays along a *new* axis**, thereby *increasing the rank* (number of dimensions) of the result.  NumPy provides several helpers (`np.stack`, `np.vstack`, `np.hstack`, `np.dstack`, …), but the most general is `np.stack`, which lets you insert the new axis anywhere with the `axis` argument.

#### Example: Stacking Along a New Axis

```python
import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# Vertical stacking (default axis=0)
c = np.stack((a, b))
print("Vertically stacked:\n", c)

# Horizontal stacking (axis=1)
d = np.stack((a, b), axis=1)
print("\nHorizontally stacked:\n", d)
```

Expected output:

```
Vertically stacked:
 [[[1 2]
  [3 4]]

 [[5 6]
  [7 8]]]

Horizontally stacked:
 [[[1 2]
  [5 6]]

 [[3 4]
  [7 8]]]
```

**Vertical Stacking (`axis=0`):**

- `np.stack((a, b))` stacks arrays `a` and `b` along a new first axis (axis=0).
- The resulting array `c` has a shape of `(2, 2, 2)`, indicating two stacked 2x2 matrices.
- Vertical stacking is useful when you need to combine datasets that have the same number of columns but represent different observations or samples.

**Horizontal Stacking (`axis=1`):**

- `np.stack((a, b), axis=1)` stacks arrays `a` and `b` along a new second axis (axis=1).
- The resulting array `d` also has a shape of `(2, 2, 2)`, but the stacking orientation differs, effectively pairing corresponding rows from each array.
- Horizontal stacking is beneficial when combining features from different datasets that share the same number of rows, allowing for the integration of multiple feature sets side by side.

> **Performance note.**`np.stack` makes a *copy*.  If you only need a *view* with a length‑1 axis you can often use `np.expand_dims` or slicing (`a[None, …]`).

### Concatenating Arrays

Concatenation merges arrays **along an existing axis** (so rank stays the same).  The canonical helper is `np.concatenate`.

#### Example: Concatenation Along Existing Axes

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# Vertical concatenation (default axis=0)
c = np.concatenate((a, b))
print("Vertically concatenated:\n", c)

# Horizontal concatenation (axis=1)
d = np.concatenate((a, b), axis=1)
print("\nHorizontally concatenated:\n", d)
```

Expected output:

```
Vertically concatenated:
 [[1 2]
 [3 4]
 [5 6]
 [7 8]]

Horizontally concatenated:
 [[1 2 5 6]
 [3 4 7 8]]
```

**Vertical Concatenation (`axis=0`):**

- `np.concatenate((a, b))` joins arrays `a` and `b` along the first axis (rows).
- The resulting array `c` has a shape of `(4, 2)`, effectively stacking `b` below `a`.
- Vertical concatenation is ideal for combining datasets with the same number of columns but different rows, such as appending new data samples to an existing dataset.

**Horizontal Concatenation (`axis=1`):**

- `np.concatenate((a, b), axis=1)` joins arrays `a` and `b` along the second axis (columns).
- The resulting array `d` has a shape of `(2, 4)`, placing `b` to the right of `a`.
- Horizontal concatenation is useful when merging feature sets from different sources that have the same number of observations, enabling the combination of multiple attributes into a single dataset.

> **Tip.** For lists of many equally‑shaped arrays, using `np.vstack`/`hstack` can be more expressive, but internally they call `concatenate`.

### Appending to Arrays

Appending involves adding elements or arrays to the end of an existing array. The `np.append()` function is straightforward and allows for both simple and complex append operations. It's worth to note that `np.append` is a convenience wrapper around `np.concatenate` that *defaults* to `axis=None`, meaning it first **flattens** its inputs.

#### Example: Appending Values

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Append b to a
c = np.append(a, b)
print("Appended array:\n", c)
```

Expected output:

```
Appended array:
 [1 2 3 4 5 6]
```

Explanation:

- `np.append(a, b)` concatenates array `b` to the end of array `a`, resulting in a new array `c` that combines all elements from both arrays.
- Appending is useful when you need to dynamically add new data points to an existing array, such as adding new user inputs or streaming data to a dataset.

**Additional Considerations:**

By default, `np.append()` flattens the input arrays if the axis is not specified. To append along a specific axis, you must ensure that the arrays have compatible shapes.

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])

# Append along axis=0 (rows)
c = np.append(a, b, axis=0)
print("Appended along axis 0:\n", c)

# Append along axis=1 (columns)
d = np.append(a, [[5], [6]], axis=1)
print("\nAppended along axis 1:\n", d)
```

Expected output:
  
```
Appended along axis 0:
[[1 2]
 [3 4]
 [5 6]]

Appended along axis 1:
[[1 2 5]
 [3 4 6]]
```

When specifying an axis, ensure that the dimensions other than the specified axis match between the arrays being appended.

> **When performance matters**, prefer `np.concatenate` (or pre‑allocate and fill) because `np.append` always copies data and is $O(n^2)$ if used inside loops.

### Splitting Arrays

Splitting breaks down an array into smaller subarrays. This operation is useful for dividing data into manageable chunks, preparing batches for machine learning models, or separating data into distinct groups for analysis. NumPy's `np.split()` function is commonly used for this purpose.

#### Regular and Custom Splits

Regular splits divide an array into equal parts, while custom splits allow you to specify the exact indices where the array should be divided.

```python
a = np.array([1, 2, 3, 4, 5, 6])

# Split into three equal parts
b = np.split(a, 3)
print("Regular split:\n", b)

# Split at the 2nd and 4th indices
c = np.split(a, [2, 4])
print("\nCustom split:\n", c)
```

Expected output:

```
Regular split:
 [array([1, 2]), array([3, 4]), array([5, 6])]

Custom split:
 [array([1, 2]), array([3, 4]), array([5, 6])]
```

**Regular Split (`np.split(a, 3)`):**

- Divides array `a` into three equal parts.
- Each resulting subarray has two elements.
- **Practical Use Case:** Regular splitting is useful when you need to divide data into uniform batches for parallel processing or batch training in machine learning.

**Custom Split (`np.split(a, [2, 4])`):**

- Splits array `a` at indices 2 and 4, resulting in three subarrays: the first containing elements up to index 2, the second from index 2 to 4, and the third from index 4 onwards.
- **Practical Use Case:** Custom splitting allows for flexible division of data based on specific criteria or requirements, such as separating features from labels in a dataset or dividing a dataset into training and testing sets.

**Additional Considerations:**

When performing a regular split, the array must be divisible into the specified number of sections. If it is not, NumPy will raise a `ValueError`.
  
```python
a = np.array([1, 2, 3, 4, 5])

try:
  b = np.split(a, 3)
except ValueError as e:
  print("Error:", e)
```

Expected output:

```
Error: array split does not result in an equal division
```

Depending on the specific needs, other splitting functions like `np.hsplit()`, `np.vsplit()`, and `np.dsplit()` can be used to split arrays along specific axes.

### Advanced Joining and Splitting Techniques

Beyond basic stacking, concatenation, appending, and splitting, NumPy offers additional functions that provide more control and flexibility when manipulating array structures.

#### Example: HStack and VStack

**Horizontal Stack (`hstack`):**

* Combines arrays horizontally (column-wise), aligning them side by side.
* Useful for merging features from different datasets or adding new features to an existing dataset.

```
a : (3,)  →  [1 2 3]
b : (3,)  →  [4 5 6]
────────────────────
hstack(a, b) : (6,) → [1 2 3 4 5 6]
```

**Vertical Stack (`vstack`):**

* Combines arrays vertically (row-wise), stacking them on top of each other.
* Ideal for adding new samples or observations to an existing dataset.

```
a : (3,) → [1 2 3]
b : (3,) → [4 5 6]
───────────────────
vstack(a, b) : (2, 3)
[[1 2 3]
 [4 5 6]]
```

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Horizontal stack
h_stack = np.hstack((a, b))
print("Horizontal stack:\n", h_stack)

# Vertical stack
v_stack = np.vstack((a, b))
print("\nVertical stack:\n", v_stack)
```

Expected output:

```
Horizontal stack:
 [1 2 3 4 5 6]

Vertical stack:
 [[1 2 3]
  [4 5 6]]
```

**Practical Use Cases**

| Function    | Typical Role in ML / Data Analysis         | Mathematical Analogy                                                                                     |
| ----------- | ------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| `np.hstack` | Merging multiple feature vectors into one  | Concatenating two vectors $x \in \mathbb{R}^n$ and $y \in \mathbb{R}^m$ to form $z \in \mathbb{R}^{n+m}$ |
| `np.vstack` | Adding new observations to a sample matrix | Forming a block matrix $\begin{bmatrix}A \\ B\end{bmatrix}$                                              |

**Performance Tips**

* Both wrappers call `np.concatenate` under the hood (`axis=1` for `hstack`, `axis=0` for `vstack`), so they **always allocate new memory**.
  👉 If you must stack inside a loop, collect references in a list and call one `np.concatenate` at the end to avoid repeated reallocations.
* For very large arrays, pre-allocate a destination array with `np.empty` and use slice assignment; that’s \~2-3× faster because only *one* large copy occurs.
* If you only need a *view* (e.g., to treat two 1-D vectors as a 2 × n matrix without copying), use `np.stack([a, b], axis=0)`; this often succeeds without copying if the originals are already contiguous.

#### Example: DStack

* Stacks arrays along the third axis (depth), creating a 3-D array.
* Useful for combining multiple 2-D arrays into a single 3-D array, such as adding color channels to grayscale images.

```
a : (2, 2)      b : (2, 2)
[[1 2]          [[5 6]
 [3 4]]          [7 8]]

────────────── dstack(a, b) : (2, 2, 2) ──────────────
 depth-0          depth-1
[[1 5]           [[2 6]
 [3 7]]           [4 8]]
```

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# Depth stack
d_stack = np.dstack((a, b))
print("Depth stack:\n", d_stack)
```

Expected output:

```
Depth stack:
 [[[1 5]
   [2 6]]

  [[3 7]
   [4 8]]]
```

**Explanation**

* `np.dstack((a, b))` combines `a` and `b` along a new third axis, resulting in a 3-D array where each “layer” corresponds to one of the original arrays.
* Equivalent call: `np.concatenate((a[..., None], b[..., None]), axis=2)`.

**Practical Use Cases**

| Scenario                 | How `dstack` Helps                              | Math / Signal-Processing View                             |
| ------------------------ | ----------------------------------------------- | --------------------------------------------------------- |
| RGB image construction   | Stack R, G, B grayscale layers into `(H, W, 3)` | Treats each pixel as a 3-vector $(r,g,b)^\top$            |
| Multichannel sensor data | Combine simultaneous 2-D sensor frames          | Produces a rank-3 tensor $X_{ijc}$ with channel index $c$ |

**Performance Tips**

* For many channels, consider `np.moveaxis` or keep data in shape `(C, H, W)` (channel-first) to exploit cache-friendly contiguous strides in deep-learning frameworks.
* Converting between `(H,W,C)` and `(C,H,W)` is a **view** (stride permutation) if you use `np.transpose`; no data copy is made.

### Practical Applications and Considerations

Knowing how to join and split arrays unlocks several everyday data-manipulation workflows:

1. **Data preprocessing** – Concatenate raw datasets or carve out train-validation-test splits so the data reaches your model in the right shape.
2. **Data augmentation** – Combine and slice existing samples to create synthetic variations, giving the model a richer, more diverse training set.
3. **Feature engineering** – Stitch multiple feature blocks together into a single matrix, allowing the algorithm to learn from a unified view of the data.
4. **Batch processing** – Break huge datasets into memory-friendly chunks, making large-scale computation feasible even on modest hardware.

### Summary Table

| Operation            | Method/Function  | Description (➜ perf tips)                                                                                           | Example Code                     | Example Output (+ shape)                                                    |
| -------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------- | -------------------------------- | --------------------------------------------------------------------------- |
| **Stack (new axis)** | `np.stack`       | Inserts a *new* axis and stacks along it.<br>➜ Collect in a list, call **once** to avoid repeated reallocations.    | `np.stack((A, B), axis=0)`       | `[[[1 2] [3 4]]  ← depth 0\n [[5 6] [7 8]]] ← depth 1`<br>*shape (2, 2, 2)* |
| **Horizontal stack** | `np.hstack`      | Concatenates **column-wise** (`axis=1` for ≥2-D, `axis=0` otherwise).                                               | `np.hstack((a1, b1))`            | `[1 2 3 4 5 6]`<br>*shape (6,)*                                             |
| **Vertical stack**   | `np.vstack`      | Concatenates **row-wise** (`axis=0`).                                                                               | `np.vstack((a1, b1))`            | `[[1 2 3]\n [4 5 6]]`<br>*shape (2, 3)*                                     |
| **Depth stack**      | `np.dstack`      | Adds a **third axis** (“depth”).<br>➜ Equivalent to `np.stack(..., axis=2)`.                                        | `np.dstack((A, B))`              | `[[[1 5] [2 6]]\n [[3 7] [4 8]]]`<br>*shape (2, 2, 2)*                      |
| **Concatenate**      | `np.concatenate` | Joins along an *existing* axis; no new dimension is created.                                                        | `np.concatenate((A, B), axis=0)` | `[[1 2]\n [3 4]\n [5 6]\n [7 8]]`<br>*shape (4, 2)*                         |
| **Append**           | `np.append`      | Thin wrapper around `concatenate` that **always flattens** first—handy for quick scripts, but avoid in tight loops. | `np.append(a1, b1)`              | `[1 2 3 4 5 6]`<br>*shape (6,)*                                             |
| **Split**            | `np.split`       | Splits **1-D** or **n-D** arrays at index positions; returns a list of views.                                       | `np.split(a1, [2, 4])`           | `[array([1, 2]), array([3, 4]), array([5, 6])]`                             |
| **Horizontal split** | `np.hsplit`      | Column-wise split of a 2-D array.                                                                                   | `np.hsplit(A, 2)`                | `[array([[1], [3]]), array([[2], [4]])]`                                    |
| **Vertical split**   | `np.vsplit`      | Row-wise split of a 2-D array.                                                                                      | `np.vsplit(A, 2)`                | `[array([[1, 2]]), array([[3, 4]])]`                                        |

Quick Math Connections:

**Stack vs. Concat**

$$\text{stack}: \mathbb{R}^{m\times n}\times\mathbb{R}^{m\times n}\;\to\;\mathbb{R}^{2\times m\times n}$$ 

(rank ↑)

$$\text{concatenate}: \mathbb{R}^{m\times n}\times\mathbb{R}^{m\times n}\;\to\;\mathbb{R}^{(2m)\times n}$$ 

(rank unchanged)

**Depth stacking** is the tensor equivalent of forming a block-diagonal matrix, grouping channels so later operations (e.g., convolution) can exploit separable structure.

Speed Rules of Thumb:

1. **Batch first, stack/concat once.** Repeated small calls spend most time reallocating memory.
2. **Use views when possible.** `np.stack` may avoid a copy if the input arrays are already contiguous and aligned.
3. **Mind order & alignment.** Converting to the required memory order (`C` vs. `F`) *once* at the start is faster than implicit copies later.


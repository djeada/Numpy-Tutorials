## Vectors

A vector is a mathematical entity characterized by both magnitude and direction. Vectors are essential in various fields such as linear algebra, calculus, physics, computer science, data analysis, and machine learning. In the context of NumPy, vectors are represented as one-dimensional arrays, enabling efficient computation and manipulation. This guide delves into the definition of vectors, their properties, and the operations that can be performed on them using NumPy, complemented by practical examples to illustrate each concept.

### Definitions

A **vector space** over a field $\mathbb{F}$ (here the reals $\mathbb{R}$) is a set equipped with vector addition and scalar multiplication that satisfy eight axioms (closure, associativity, identity, inverses, distributive laws, etc.).  The canonical example is the *n‑dimensional real coordinate space* $\mathbb{R}^n$.

### Vector in $\mathbb{R}^n$

*Formal definition.*  An element $\mathbf v \in \mathbb{R}^n$ is an ordered $n$-tuple of real numbers

$$
  \mathbf v = (v_1,\dots,v_n) \equiv \sum_{i=1}^n v_i\mathbf e_i
$$

where $\{\mathbf e_i\}_{i=1}^n$ is the standard basis with $\mathbf e_i$ having a *1* in the $i$-th position and zeros elsewhere.

A vector encodes **magnitude** and **direction** relative to the origin.  In data‑science terms, it stores the *feature values* of one sample.

*NumPy quick‑start.*

```python
import numpy as np
v = np.array([4, -2, 7])  # element of R^3
type(v), v.shape          # (numpy.ndarray, (3,))
```

#### Row vs Column Representation

Vectors can be represented in two distinct forms: row vectors and column vectors. The orientation of a vector affects how it interacts with matrices and other vectors during mathematical operations.

**Row Vector:**

A row vector displays its elements horizontally and is represented as a $1 \times n$ matrix.
A row vector is written as a $1 \times n$ matrix, such as $\begin{bmatrix} 1 & 2 & 3 \end{bmatrix}$.

**Example:**

$\vec{v} = [1, 2, 3]$ is a row vector in $\mathbb{R}^3$.

Row vectors are commonly used to represent individual data samples in a dataset, where each element corresponds to a feature value.

**Column Vector:**

A column vector displays its elements vertically and is represented as an $n \times 1$ matrix.

A column vector is written as an $n \times 1$ matrix, such as:

$$
\begin{bmatrix}
1 \\
2 \\
3 
\end{bmatrix}
$$

Column vectors are used in linear transformations and matrix operations, where they can be multiplied by matrices to perform operations like rotations and scaling.

#### Transpose

The transpose of a vector changes its orientation from a row vector to a column vector or vice versa. Transposing is a fundamental operation in linear algebra, facilitating various matrix and vector computations.

- Transposing a vector converts a row vector to a column vector and a column vector to a row vector.
- The transpose of a vector $\vec{v}$ is denoted as $\vec{v}^T$.
- If $\vec{v} = [1, 2, 3]$ (a row vector), then $\vec{v}^T$ is:

$$
\vec{v}^T = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}
$$

- Transposing vectors is essential when performing dot products between row and column vectors or when aligning data structures for matrix multiplication.

#### Norms and Length

A **norm** $||·||$ assigns a non‑negative length to every vector and obeys positivity, scalability, and the triangle inequality.

**p-Norm:**

$$
||\vec{v}||_p = \left( \sum_{i=1}^{n} |v_i|^p \right)^{1/p}
$$

- The $p$-norm generalizes different measures of vector length.
- The most commonly used norm is the L2 norm (Euclidean norm), where $p=2$:

$$
||\vec{v}||_2 = \sqrt{v_1^2 + v_2^2 + \ldots + v_n^2}
$$

Special cases:

| p | Common name | Unit ball in $\mathbb{R}^2$ |
| - | ----------- | --------------------------- |
| 1 | Manhattan   | diamond ‑◆‑                 |
| 2 | Euclidean   | circle ‑◎‑                  |
| ∞ | Chebyshev   | square  ▢                   |

Sketches (unit radius):

![image](https://github.com/user-attachments/assets/ed74c48f-7af6-47d7-988e-5c6fae7788fd)

NumPy examples:

```python
from numpy.linalg import norm
norm(v, ord=1)   # L1
norm(v)          # default ord=2
norm(v, ord=np.inf)
```

*Why it matters.*  In machine‑learning metrics (e.g. k‑NN), the norm defines "closeness"; in optimization, the choice of norm shapes the feasible region and affects sparsity.

### Vector Operations

NumPy provides a comprehensive suite of functions to perform various vector operations efficiently. These operations are essential for tasks in data analysis, machine learning, physics simulations, and more.

#### Vector Addition

Vector addition combines two vectors by adding their corresponding elements, resulting in a new vector. This operation is fundamental in many applications, including physics for calculating resultant forces.

```python
import numpy as np

arr_1 = np.array([9, 2, 5])
arr_2 = np.array([-3, 8, 2])

# Element-wise addition
result = np.add(arr_1, arr_2)
print(result)
```

Expected output:

```
[ 6 10  7]
```

- `np.add(arr_1, arr_2)` performs element-wise addition of `arr_1` and `arr_2`.
- The resulting vector `[6, 10, 7]` is obtained by adding each corresponding pair of elements: $9 + (-3) = 6$, $2 + 8 = 10$, and $5 + 2 = 7$.
- Vector addition is used in aggregating multiple data sources, such as combining different feature sets in data preprocessing.

#### Scalar Multiplication

Scalar multiplication involves multiplying each element of a vector by a scalar (a single numerical value), effectively scaling the vector's magnitude without altering its direction.

```python
arr = np.array([6, 3, 4])
scalar = 2

# Scalar multiplication
result = scalar * arr
print(result)
```

Expected output:

```
[12  6  8]
```

- `scalar * arr` multiplies each element of `arr` by `2`.
- The resulting vector `[12, 6, 8]` reflects the scaled values.
- Scalar multiplication is used in normalization processes, where feature values are scaled to a specific range to ensure uniformity.

#### Dot Product

The dot product of two vectors yields a scalar value and is calculated as the sum of the products of their corresponding elements. It measures the cosine of the angle between two vectors and is widely used in various applications, including calculating projections and in machine learning algorithms.

```python
arr_1 = np.array([9, 2, 5])
arr_2 = np.array([-3, 8, 2])

# Dot product
result = np.dot(arr_1, arr_2)
print(result)
```

Expected output:

```
-1
```

- `np.dot(arr_1, arr_2)` computes the dot product: $9*(-3) + 2*8 + 5*2 = -27 + 16 + 10 = -1$.
- The result `-1` is a scalar indicating the degree of similarity between the two vectors.
- Dot products are used in calculating the similarity between two data points in recommendation systems and in determining the direction of force vectors in physics.

#### Cross Product

The cross product is defined for three-dimensional vectors and results in a new vector that is perpendicular to both input vectors. It is particularly useful in physics and engineering for finding torque or rotational forces.

```python
arr_1 = np.array([9, 2, 5])
arr_2 = np.array([-3, 8, 2])

# Cross product
result = np.cross(arr_1, arr_2)
print(result)
```

Expected output:

```
[-36 -33  78]
```

Explanation:

- `np.cross(arr_1, arr_2)` calculates the cross product using the formula:

$$\vec{v} \times \vec{w} = \begin{bmatrix}
v_2w_3 - v_3w_2 \\
v_3w_1 - v_1w_3 \\
v_1w_2 - v_2w_1 
\end{bmatrix}$$

- For the given vectors:

$$\begin{aligned}
x &= 2*2 - 5*8 = 4 - 40 = -36 \\
y &= 5*(-3) - 9*2 = -15 - 18 = -33 \\
z &= 9*8 - 2*(-3) = 72 + 6 = 78 
\end{aligned}$$

- The resulting vector `[-36, -33, 78]` is perpendicular to both `arr_1` and `arr_2`.
- Cross products are used in computer graphics to calculate surface normals, which are essential for rendering lighting and shading effects.

#### Angle Between Vectors

The angle between two vectors can be determined using the dot product and the vectors' norms. This angle provides insight into the relationship between the vectors, such as their alignment or orthogonality.

```arr_1 = np.array([9, 2, 5])
arr_2 = np.array([-3, 8, 2])

# Angle between vectors
cos_angle = np.dot(arr_1, arr_2) / (np.linalg.norm(arr_1) * np.linalg.norm(arr_2))
angle_rad = np.arccos(cos_angle)
print(angle_rad)
```

Expected output:

```
1.582
```

Explanation:

- `np.dot(arr_1, arr_2)` calculates the dot product of the two vectors.
- `np.linalg.norm(arr_1)` and `np.linalg.norm(arr_2)` compute the L2 norms (Euclidean lengths) of the vectors.
- The cosine of the angle is obtained by dividing the dot product by the product of the norms.
- `np.arccos(cos_angle)` computes the angle in radians.
- The resulting angle `1.582` radians indicates the measure between the two vectors.
- Calculating angles between vectors is essential in machine learning for determining feature importance and in physics for understanding force directions.

### Broadcasting

Broadcasting is a powerful feature in NumPy that allows arithmetic operations on arrays of different shapes and sizes without explicit replication of data. It simplifies code and enhances performance by enabling vectorized operations.

#### Example of Broadcasting

Broadcasting automatically expands the smaller array to match the shape of the larger array during arithmetic operations. This feature eliminates the need for manual looping and ensures efficient computation.

```arr = np.array([1, 2, 3, 4])
scalar = 2

# Broadcasting operations
print("Addition with scalar:", arr + scalar)
print("Multiplication with scalar:", arr * scalar)
```

Expected output:

```
Addition with scalar: [3 4 5 6]
Multiplication with scalar: [2 4 6 8]
```

Explanation:

- `arr + scalar` adds `2` to each element of `arr`, resulting in `[3, 4, 5, 6]`.
- `arr * scalar` multiplies each element of `arr` by `2`, resulting in `[2, 4, 6, 8]`.
- NumPy automatically broadcasts the scalar to match the shape of the array for element-wise operations.
- Broadcasting is used in data normalization, where a mean vector is subtracted from a dataset, and in scaling features by multiplying with a scalar value to adjust their range.

### Practical Applications

Vectors and their operations are integral to numerous practical applications across various domains. Mastering these concepts enables efficient data manipulation, analysis, and the implementation of complex algorithms.

#### Accessing and Modifying Multiple Elements

Beyond single-element access, vectors allow for the manipulation of multiple elements simultaneously using slicing or advanced indexing. This capability is essential for batch processing and data transformation tasks.

```# Creating a 1D array
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# Modifying multiple elements
arr[2:5] = [10, 11, 12]
print(arr)
```

Expected output:

```
[ 1  2 10 11 12  6  7  8]
```

- `arr[2:5] = [10, 11, 12]` assigns the values `10`, `11`, and `12` to the elements at indices `2`, `3`, and `4`, respectively.
- The original array `[1, 2, 3, 4, 5, 6, 7, 8]` is updated to `[1, 2, 10, 11, 12, 6, 7, 8]`.
- Batch updating is useful in data cleaning processes where multiple data points need correction or transformation, such as replacing outliers or applying scaling factors to specific sections of a dataset.

#### Boolean Indexing

Boolean indexing enables the selection of elements based on conditional statements, allowing for dynamic and flexible data selection without the need for explicit loops. This technique is highly efficient and widely used in data analysis.

```# Creating a 1D array
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
# Boolean indexing
bool_idx = arr > 5
print(arr[bool_idx])
```

Expected output:

```
[6 7 8]
```

- `arr > 5` creates a boolean array `[False, False, False, False, False, True, True, True]`.
- `arr[bool_idx]` uses this boolean array to filter and retrieve elements where the condition `arr > 5` is `True`, resulting in `[6, 7, 8]`.
- Boolean indexing is used to filter datasets based on specific criteria, such as selecting all records where a sales figure exceeds a certain threshold or extracting all entries that meet particular quality standards.

### Summary Table
All examples are *self-contained*—each row declares the minimal variables it needs—so you can copy-paste any cell directly into any IDE.

| Operation                 | Description & Formula                                                                                            | Example Code                                                                                                                                                                                                                                                  | Expected Output (shape)  |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| **Vector Addition**       | Element-wise sum<br> $c_i = a_i + b_i$                                                                           | `arr_1 = np.array([9, 2, 5])`<br>`arr_2 = np.array([-3, 8, 2])`<br>`np.add(arr_1, arr_2)`                                                                                                                                | `[ 6 10  7]` `(3,)`      |
| **Scalar Multiplication** | Scale a vector<br> $c_i = ka_i$                                                                                | `scalar = 2`<br>`arr = np.array([6, 3, 4])`<br>`scalar * arr`                                                                                                                                                                                                 | `[12  6  8]` `(3,)`      |
| **Dot Product**           | Projection / cosine similarity<br> $a \cdot b = \sum_i a_i b_i$                                                  | `arr_1 = np.array([9, 2, 5])`<br>`arr_2 = np.array([-3, 8, 2])`<br>`np.dot(arr_1, arr_2)`                                                                                                                            | `-1` `()`                |
| **Cross Product**         | 3-D vector orthogonal to both inputs<br> $a \times b$                                                            | `arr_1 = np.array([9, 2, 5])`<br>`arr_2 = np.array([-3, 8, 2])`<br>`np.cross(arr_1, arr_2)`                                                                                                                          | `[-36 -33  78]` `(3,)`   |
| **Angle Between Vectors** | $\theta = \arccos\!\left(\dfrac{a\cdot b}{\|a\|\|b\|}\right)$                                                    | `arr_1 = np.array([9, 2, 5])`<br>`arr_2 = np.array([-3, 8, 2])`<br>`angle = np.arccos(np.dot(arr_1, arr_2) / (np.linalg.norm(arr_1)*np.linalg.norm(arr_2)))`<br>`np.round(angle, 3)`                                   | `1.582` rad              |
| **Broadcasting**          | NumPy automatically “stretches” smaller shapes so element-wise ops make sense.<br>*(vector ⇄ scalar shown here)* | `arr = np.array([1, 2, 3, 4])`<br>`scalar = 2`<br>`arr + scalar, arr * scalar`                                                                                                                                           | `([3 4 5 6], [2 4 6 8])` |

Tiny Performance Tips:

* **Vectorized > loops** – every row above is a single, optimized C call.
* **`np.dot` & BLAS** – use contiguous `float64` arrays for best throughput.
* **Broadcast with care** – repeated implicit copies are *virtual*, but an unexpected `np.copy()` downstream can explode memory; check `arr.strides`.

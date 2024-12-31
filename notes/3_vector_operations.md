## Vectors

A vector is a fundamental mathematical entity characterized by both magnitude and direction. Vectors are essential in various fields such as linear algebra, calculus, physics, computer science, data analysis, and machine learning. In the context of NumPy, vectors are represented as one-dimensional arrays, enabling efficient computation and manipulation. This guide delves into the definition of vectors, their properties, and the operations that can be performed on them using NumPy, complemented by practical examples to illustrate each concept.

### Vector Definitions

Understanding the basic definitions and properties of vectors is crucial for performing effective computations and leveraging their capabilities in various applications.

#### Vector in $\mathbb{R}^n$

A vector in $\mathbb{R}^n$ is an ordered collection of $n$ real numbers, where each number represents a component of the vector. This $n$-tuple of real numbers defines both the magnitude and direction of the vector in an $n$-dimensional space.


A vector $\vec{v}$ in $\mathbb{R}^n$ is expressed as $\vec{v} = (v_1, v_2, \ldots, v_n)$, where each $v_i$ is a real number.

**Example:**

In $\mathbb{R}^3$, a vector can be represented as $\vec{v} = (4, -2, 7)$.

**Practical Use Case:** Vectors in $\mathbb{R}^n$ are used to represent data points in machine learning, where each component corresponds to a feature of the data.

#### Row vs. Column Vectors

Vectors can be represented in two distinct forms: row vectors and column vectors. The orientation of a vector affects how it interacts with matrices and other vectors during mathematical operations.

**Row Vector:**

A row vector displays its elements horizontally and is represented as a $1 \times n$ matrix.
A row vector is written as a $1 \times n$ matrix, such as $\begin{bmatrix} 1 & 2 & 3 \end{bmatrix}$.

**Example:**

$\vec{v} = [1, 2, 3]$ is a row vector in $\mathbb{R}^3$.

**Practical Use Case:** Row vectors are commonly used to represent individual data samples in a dataset, where each element corresponds to a feature value.

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

**Practical Use Case:** Column vectors are used in linear transformations and matrix operations, where they can be multiplied by matrices to perform operations like rotations and scaling.

#### Transpose

The transpose of a vector changes its orientation from a row vector to a column vector or vice versa. Transposing is a fundamental operation in linear algebra, facilitating various matrix and vector computations.

- Transposing a vector converts a row vector to a column vector and a column vector to a row vector.
- The transpose of a vector $\vec{v}$ is denoted as $\vec{v}^T$.
- If $\vec{v} = [1, 2, 3]$ (a row vector), then $\vec{v}^T$ is:

$$
\vec{v}^T = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}
$$

- **Practical Use Case:** Transposing vectors is essential when performing dot products between row and column vectors or when aligning data structures for matrix multiplication.

#### Norm

The norm of a vector quantifies its magnitude or length. Various norms can be used depending on the context, each providing a different measure of the vector's size.

- The norm of a vector $\vec{v}$ measures its length in the vector space.
- **p-Norm:**

$$
||\vec{v}||_p = \left( \sum_{i=1}^{n} |v_i|^p \right)^{1/p}
$$

- The $p$-norm generalizes different measures of vector length.
- The most commonly used norm is the L2 norm (Euclidean norm), where $p=2$:

$$
||\vec{v}||_2 = \sqrt{v_1^2 + v_2^2 + \ldots + v_n^2}
$$

- **Practical Use Case:** Norms are used in machine learning algorithms to measure distances between data points, such as in k-nearest neighbors (KNN) and support vector machines (SVM).

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

Explanation:

- `np.add(arr_1, arr_2)` performs element-wise addition of `arr_1` and `arr_2`.
- The resulting vector `[6, 10, 7]` is obtained by adding each corresponding pair of elements: $9 + (-3) = 6$, $2 + 8 = 10$, and $5 + 2 = 7$.
- **Practical Use Case:** Vector addition is used in aggregating multiple data sources, such as combining different feature sets in data preprocessing.
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

Explanation:

- `scalar * arr` multiplies each element of `arr` by `2`.
- The resulting vector `[12, 6, 8]` reflects the scaled values.
- **Practical Use Case:** Scalar multiplication is used in normalization processes, where feature values are scaled to a specific range to ensure uniformity.

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

Explanation:

- `np.dot(arr_1, arr_2)` computes the dot product: $9*(-3) + 2*8 + 5*2 = -27 + 16 + 10 = -1$.
- The result `-1` is a scalar indicating the degree of similarity between the two vectors.
- **Practical Use Case:** Dot products are used in calculating the similarity between two data points in recommendation systems and in determining the direction of force vectors in physics.
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
- **Practical Use Case:** Cross products are used in computer graphics to calculate surface normals, which are essential for rendering lighting and shading effects.

#### Angle Between Vectors

The angle between two vectors can be determined using the dot product and the vectors' norms. This angle provides insight into the relationship between the vectors, such as their alignment or orthogonality.

```python
arr_1 = np.array([9, 2, 5])
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
- **Practical Use Case:** Calculating angles between vectors is essential in machine learning for determining feature importance and in physics for understanding force directions.

### Broadcasting

Broadcasting is a powerful feature in NumPy that allows arithmetic operations on arrays of different shapes and sizes without explicit replication of data. It simplifies code and enhances performance by enabling vectorized operations.

#### Example of Broadcasting

Broadcasting automatically expands the smaller array to match the shape of the larger array during arithmetic operations. This feature eliminates the need for manual looping and ensures efficient computation.

```python
arr = np.array([1, 2, 3, 4])
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
- **Practical Use Case:** Broadcasting is used in data normalization, where a mean vector is subtracted from a dataset, and in scaling features by multiplying with a scalar value to adjust their range.

### Practical Applications

Vectors and their operations are integral to numerous practical applications across various domains. Mastering these concepts enables efficient data manipulation, analysis, and the implementation of complex algorithms.

#### Accessing and Modifying Multiple Elements

Beyond single-element access, vectors allow for the manipulation of multiple elements simultaneously using slicing or advanced indexing. This capability is essential for batch processing and data transformation tasks.

```python
# Creating a 1D array
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# Modifying multiple elements
arr[2:5] = [10, 11, 12]
print(arr)
```

Expected output:

```
[ 1  2 10 11 12  6  7  8]
```

Explanation:

- `arr[2:5] = [10, 11, 12]` assigns the values `10`, `11`, and `12` to the elements at indices `2`, `3`, and `4`, respectively.
- The original array `[1, 2, 3, 4, 5, 6, 7, 8]` is updated to `[1, 2, 10, 11, 12, 6, 7, 8]`.
- **Practical Use Case:** Batch updating is useful in data cleaning processes where multiple data points need correction or transformation, such as replacing outliers or applying scaling factors to specific sections of a dataset.

#### Boolean Indexing

Boolean indexing enables the selection of elements based on conditional statements, allowing for dynamic and flexible data selection without the need for explicit loops. This technique is highly efficient and widely used in data analysis.

```python
# Creating a 1D array
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
# Boolean indexing
bool_idx = arr > 5
print(arr[bool_idx])
```

Expected output:

```
[6 7 8]
```

Explanation:

- `arr > 5` creates a boolean array `[False, False, False, False, False, True, True, True]`.
- `arr[bool_idx]` uses this boolean array to filter and retrieve elements where the condition `arr > 5` is `True`, resulting in `[6, 7, 8]`.
- **Practical Use Case:** Boolean indexing is used to filter datasets based on specific criteria, such as selecting all records where a sales figure exceeds a certain threshold or extracting all entries that meet particular quality standards.

### Summary Table

| Operation                | Description                                                                | Example Code                                      | Expected Output                               |
|--------------------------|----------------------------------------------------------------------------|--------------------------------------------------|----------------------------------------------|
| **Vector Addition**      | Adds two vectors element-wise.                                             | `np.add(arr_1, arr_2)`                           | `[ 6 10  7]`                                 |
| **Scalar Multiplication**| Multiplies each element of the vector by a scalar.                          | `scalar * arr`                                   | `[12  6  8]`                                 |
| **Dot Product**          | Computes the dot product of two vectors, resulting in a scalar.             | `np.dot(arr_1, arr_2)`                           | `-1`                                         |
| **Cross Product**        | Computes the cross product of two 3D vectors, resulting in a perpendicular vector. | `np.cross(arr_1, arr_2)`                         | `[-36 -33  78]`                              |
| **Angle Between Vectors**| Calculates the angle between two vectors using dot product and norms.       | `np.arccos(np.dot(arr_1, arr_2) / (np.linalg.norm(arr_1) * np.linalg.norm(arr_2)))` | `1.582`                                      |
| **Broadcasting**         | Allows arithmetic operations on arrays of different shapes.                | `arr + scalar`, `arr * scalar`                   | `Addition with scalar: [3 4 5 6]`, `Multiplication with scalar: [2 4 6 8]` |

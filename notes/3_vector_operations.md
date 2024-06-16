## Vectors in Context of NumPy

A vector is a mathematical object with both magnitude and direction, essential in linear algebra and calculus. In computer science, vectors are used for various operations in data analysis, machine learning, and scientific computing. This guide explores vectors in the context of NumPy, providing definitions, operations, and practical examples.

### Vector Definitions

#### Vector in $\mathbb{R}^n$

- A vector in $\mathbb{R}^n$ is an n-tuple of real numbers.
- It is an ordered collection of 'n' numbers, where each number is a component of the vector.
- Notation: A vector $\vec{v}$ in $\mathbb{R}^n$ is often written as $\vec{v} = (v_1, v_2, \ldots, v_n)$.

#### Row vs. Column Vectors

**Row Vector**:

- Displays the elements of the vector horizontally.
- Example: $\vec{v} = [1, 2, 3]$.
- Notation: A row vector is written as a 1xN matrix.

**Column Vector**:

- Displays the elements vertically.
- Example:

$$
\vec{v} = \begin{bmatrix}
1 \\
2 \\
3 
\end{bmatrix}
$$

- Notation: A column vector is written as an Nx1 matrix.

#### Transpose

- Transposing a vector converts a row vector to a column vector and vice versa.
- Notation: The transpose of a vector $\vec{v}$ is denoted $\vec{v}^T$.
- Example: If $\vec{v} = [1, 2, 3]$, then $\vec{v}^T = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$.

#### Norm
- The norm of a vector measures its "length" or magnitude.
- Different norms can be used depending on the context.
- **p-Norm**:

$$
||\vec{v}||_p = \left( \sum_i |v_i|^p \right)^{1/p}
$$

- The most commonly used norm is the L2 norm (Euclidean norm), where $p=2$.

### Vector Operations

NumPy supports various vector operations, including addition, scalar multiplication, dot product, cross product, and more.

#### Vector Addition

Vectors can be added element-wise to yield a new vector.

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
- `np.add(arr_1, arr_2)` performs element-wise addition.

#### Scalar Multiplication

Each element of a vector is multiplied by a scalar value.

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
- `scalar * arr` multiplies each element by 2.

#### Dot Product

The dot product of two vectors results in a scalar, calculated as the sum of the products of corresponding elements.

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
- `np.dot(arr_1, arr_2)` computes the dot product.

#### Cross Product

The cross product is defined for 3D vectors and results in a vector perpendicular to both input vectors.

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
- `np.cross(arr_1, arr_2)` computes the cross product of the two vectors.

#### Angle Between Vectors

Using the dot product and norms, the cosine of the angle between two vectors can be found. This can be inverted to get the angle.

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
- `np.arccos(np.dot(arr_1, arr_2) / (np.linalg.norm(arr_1) * np.linalg.norm(arr_2)))` calculates the angle in radians.

### Broadcasting

NumPy's broadcasting feature allows for arithmetic operations on arrays of different shapes, facilitating vectorized operations and eliminating the need for explicit loops.

#### Example of Broadcasting

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
- `arr + scalar` and `arr * scalar` apply the scalar to each element of the array.

### Summary Table

| Operation                | Description                                                                | Example Code                                      | Expected Output                               |
|--------------------------|----------------------------------------------------------------------------|--------------------------------------------------|----------------------------------------------|
| **Vector Addition**      | Adds two vectors element-wise.                                             | `np.add(arr_1, arr_2)`                           | `[ 6 10  7]`                                 |
| **Scalar Multiplication**| Multiplies each element of the vector by a scalar.                          | `scalar * arr`                                   | `[12  6  8]`                                 |
| **Dot Product**          | Computes the dot product of two vectors, resulting in a scalar.             | `np.dot(arr_1, arr_2)`                           | `-1`                                         |
| **Cross Product**        | Computes the cross product of two 3D vectors, resulting in a perpendicular vector. | `np.cross(arr_1, arr_2)`                         | `[-36 -33  78]`                              |
| **Angle Between Vectors**| Calculates the angle between two vectors using dot product and norms.       | `np.arccos(np.dot(arr_1, arr_2) / (np.linalg.norm(arr_1) * np.linalg.norm(arr_2)))` | `1.582`                                      |
| **Broadcasting**         | Allows arithmetic operations on arrays of different shapes.                | `arr + scalar`, `arr * scalar`                   | `Addition with scalar: [3 4 5 6]`, `Multiplication with scalar: [2 4 6 8]` |

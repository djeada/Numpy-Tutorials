# Vectors in Context of Numpy

A vector is a mathematical object with magnitude and direction, and it's a cornerstone of linear algebra and calculus. Here, we'll delve deeper into the specifics of vectors in the context of computer science and their applications.

## Vector Definitions

- **Vector in $R^n$**: A vector in $R^n$ is an n-tuple of real numbers. Essentially, it's an ordered collection of 'n' numbers.

- **Row vs. Column Vectors**: 
  - A `row vector` writes the elements of the vector horizontally.
  - A `column vector` writes the elements vertically.
  - Example:
    - Row vector: [1 2 3]
    - Column vector:
   
$$
\begin{bmatrix}
1 \\
2 \\
3 
\end{bmatrix}
$$

- **Transpose**: 
  - A column vector's `transpose` converts it to a row vector of the same length, and vice-versa.
  
- **Norm**: 
  - A measure of a vector's "length" or magnitude. Depending on the metric used, there are different ways to calculate this length.
  - The formula you provided is for the p-norm:
    
    $$||\vec{v}||_p = \sqrt[p]{\sum_i v_i^p}$$
    
  - The most commonly used norm is the L2 norm (or Euclidean norm), where p=2.

## Vector Operations

There are number of vector operations supported by numpy.

### Vector Addition

Vectors can be added together to yield a new vector. The addition is performed element-wise.

```python
import numpy as np
arr_1 = np.array([9, 2, 5])
arr_2 = np.array([-3, 8, 2])
print(np.add(arr_1, arr_2))
```

Expected output:

```
[ 6 10  7]
```

### Scalar Multiplication

Every element of the vector gets multiplied by the scalar.

```python
arr = np.array([6, 3, 4])
scalar = 2
print(scalar * arr)
```

Expected output:

```
[12  6  8]
```

### Dot Product

The dot product of two vectors results in a scalar. It's the sum of the product of the corresponding entries of the two sequences of numbers.

```python
arr_1 = np.array([9, 2, 5])
arr_2 = np.array([-3, 8, 2])
print(np.dot(arr_1, arr_2))
```

Expected output:

```
-1
```

### Cross Product

This operation is only defined in 3D and 7D. The result is a vector that's perpendicular to both of the original vectors.

```python
arr_1 = np.array([9, 2, 5])
arr_2 = np.array([-3, 8, 2])
print(np.cross(arr_1, arr_2))
```

Expected output:

```
[-36 -33  78]
```

### Angle between Vectors

Using dot product and norms, the cosine of the angle between two vectors can be found. This can be inverted to get the angle itself.

```python
arr_1 = np.array([9, 2, 5])
arr_2 = np.array([-3, 8, 2])
angle_rad = np.arccos(np.dot(arr_1, arr_2) / (np.linalg.norm(arr_1) * np.linalg.norm(arr_2)))
print(angle_rad)
```

Expected output:

```
1.582
```

### Broadcasting

NumPy's broadcasting feature allows arithmetic operations to be performed on arrays of different shapes, making it possible to vectorize operations and avoid explicit loops. This is particularly useful for operations involving a scalar and a vector, or arrays of compatible shapes.

#### Example:

```python
import numpy as np

arr = np.array([1, 2, 3, 4])
scalar = 2

print("Addition with scalar:", arr + scalar)
print("Multiplication with scalar:", arr * scalar)
```

Expected output:

```
Addition with scalar: [3 4 5 6]
Multiplication with scalar: [2 4 6 8]
```


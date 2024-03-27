# Matrices in Context of Numpy

A matrix is a systematic arrangement of numbers (or elements) in rows and columns. An m × n matrix has `m` rows and `n` columns. The dimensions of the matrix are represented as m × n.

## Matrix Norm 

The matrix norm is a function that measures the size or magnitude of a matrix. Just as the absolute value of a number provides its magnitude, the norm gives the magnitude of a matrix. There are various ways to define matrix norms, with each having specific properties and applications.

### Vector Norm and Matrix Norm

A vector norm is a function that assigns a non-negative value to a vector in an n-dimensional space, providing a quantitative measure of the vector's length or magnitude. A commonly used vector norm is the **Euclidean norm**, defined for a vector $\vec{x}$ in $\mathbb{R}^n$ as:

$$
||\vec{x}||_2 = \sqrt{\sum_{i=1}^n x_i^2}
$$

where $x_i$represents the components of the vector $\vec{x}$.

In the context of matrices, the **Frobenius norm** is analogous to the Euclidean norm for vectors, but it applies to matrices. For a matrix $M$with dimensions $m \times n$, the Frobenius norm is defined as:

$$
||M||_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n (M_{ij})^2}
$$

where $M_{ij}$ represents the elements of the matrix $M$. This norm can be seen as the Euclidean norm of the matrix when it is considered as a vector of its elements.

### Norms in NumPy

With NumPy, you can compute various matrix norms:

- **Frobenius norm**:

    ```Python
    import numpy as np
    A = np.array([[1, 2], [3, 4]])
    norm_A = np.linalg.norm(A, 'fro')
    print(norm_A)
    ```

    Expected output: `5.477225575051661`

- **Spectral norm**:

    ```Python
    norm_A = np.linalg.norm(A, 2)
    print(norm_A)
    ```

    Expected output: `5.464985704219043`

### Sub-multiplicative Property

Matrix norms have the sub-multiplicative property:

$$
||AB|| \leq ||A||\times||B||
$$

This characteristic implies that the norm of the matrix product doesn't exceed the product of their norms. This property is paramount in many numerical analyses.

Additionally, the matrix norm enables the definition of a distance metric between matrices:

$$
d(A,B) = ||A-B||
$$

## Matrix Multiplication
Matrix multiplication is a fundamental operation in linear algebra where the number of columns in the first matrix must match the number of rows in the second matrix. For matrices $M$ and $N$, where $M$ is an $m \times n$ matrix and $N$ is an $n \times p$ matrix, their product will be an $m \times p$ matrix. The elements of the resulting matrix are computed as follows:

$$ 
(M \times N)_{ij} = \sum_{k=1}^n M_{ik}N_{kj} 
$$

where $(M \times N)_{ij}$ is the element in the $i$-th row and $j$-th column of the resulting matrix, $M_{ik}$ is the element in the $i$-th row and $k$-th column of matrix $M$, and $N_{kj}$ is the element in the $k$-th row and $j$-th column of matrix $N$.

Here's an example of matrix multiplication using NumPy:

```Python
M = np.array([[-4, 5], [1, 7], [8, 3]])
N = np.array([[3, -5, 2, 7], [-5, 1, -4, -3]])
print(np.dot(M, N))
```

Expected output:

```
[[-37  25 -28 -43]
 [-32   2 -26 -14]
 [  9 -37   4  47]]
```

## Matrix Transpose

Transposing a matrix involves interchanging its rows and columns. The transpose of an m × n matrix results in an n × m matrix:

```Python
M = np.array([[-4, 5], [1, 7], [8, 3]])
print(M.T)
```

Expected output:

```
[[-4  1  8]
 [ 5  7  3]]
```

## Determinants

Determinants are exclusive to square matrices and hold geometric and algebraic significance:

```Python
M = np.array([[-4, 5], [1, 7]])
print(np.linalg.det(M))
```

Expected output: `-33.0`

## Identity and Inverse Matrices

The identity matrix, typically denoted as $I$, plays a crucial role in matrix algebra. It is a square matrix with ones on its main diagonal and zeros in all other positions. The identity matrix acts as the multiplicative identity in matrix operations, similar to the number 1 in scalar multiplication.

When a square matrix $A$ has an inverse, denoted as $A^{-1}$, it means that when $A$ is multiplied by $A^{-1}$, the result is the identity matrix $I$. The relationship is given by:

$$
A \times A^{-1} = A^{-1} \times A = I
$$

This property is fundamental in linear algebra, indicating that multiplying a matrix by its inverse yields the identity matrix.

```Python
M = np.array([[-4, 5], [1, 7]])
print(np.linalg.inv(M))
```

Expected output:

```
[[-0.21212121  0.15151515]
 [ 0.03030303  0.12121212]]
```

## Rank of a Matrix

The rank of a matrix provides insight into its structure and properties. Essentially, it is the number of linearly independent rows or columns present in the matrix. The rank can reveal information about the solutions of linear systems or the invertibility of a matrix.

### Understanding Matrix Rank

- **Linear Independence**: Rows (or columns) are linearly independent if no row (or column) can be expressed as a linear combination of others.
  
- **Full Rank**: A matrix is considered to have full rank if its rank is equal to the lesser of its number of rows and columns. A matrix with full rank has the maximum possible rank given its dimensions.

### Determining the Rank with NumPy

Python's NumPy library offers a convenient function to compute the rank: `np.linalg.matrix_rank`.

Example:

```Python
import numpy as np

# Define a matrix
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Calculate its rank
rank_A = np.linalg.matrix_rank(A)

print("Rank of A:", rank_A)
```

Output:

```
Rank of A: 2
```

In this instance, the rank of matrix A is 2, suggesting that only 2 of its rows (or columns) are linearly independent.

### Singular Matrices and Rank

A matrix's rank can indicate whether it's singular (non-invertible). A square matrix is singular if its rank is less than its size (number of rows or columns). Singular matrices don't possess unique inverses.

To check for singularity using rank:

```Python
import numpy as np

# Create a matrix, which is clearly singular due to linearly dependent rows
A = np.array([[1, 2], [2, 4]])

# Calculate the rank
rank_A = np.linalg.matrix_rank(A)

# Check for singularity
is_singular = "Matrix A is singular." if rank_A < A.shape[1] else "Matrix A is not singular."

print(is_singular)
```

Output:

```
Matrix A is singular.
```

By understanding the rank, one can determine the properties of a matrix and its ability to be inverted, which is crucial in numerous linear algebra applications.

## Summary of Matrix and Vector Operations

In NumPy, element-wise operations are performed on arrays (vectors and matrices) where corresponding elements are processed together. Here are some common element-wise operations:

| Operation     | NumPy Function             | Python Operator |
|---------------|----------------------------|-----------------|
| Addition      | `np.add(arr_1, arr_2)`     | `arr_1 + arr_2` |
| Subtraction   | `np.subtract(arr_1, arr_2)`| `arr_1 - arr_2` |
| Multiplication| `np.multiply(arr_1, arr_2)`| `arr_1 * arr_2` |
| Division      | `np.divide(arr_1, arr_2)`  | `arr_1 / arr_2` |

The Python operators (`+`, `-`, `*`, `/`) can be used for convenience and are equivalent to their respective NumPy functions.

Here's an example demonstrating these operations using NumPy arrays:

```Python
import numpy as np

arr_1 = np.array([1, 2, 3, 4])
arr_2 = np.array([1, 2, 3, 4])

print("Addition:", arr_1 + arr_2)
print("Subtraction:", arr_1 - arr_2)
print("Multiplication:", arr_1 * arr_2)
print("Division:", arr_1 / arr_2)
```

Expected output:

```
Addition: [2 4 6 8]
Subtraction: [0 0 0 0]
Multiplication: [1 4 9 16]
Division: [1. 1. 1. 1.]
```

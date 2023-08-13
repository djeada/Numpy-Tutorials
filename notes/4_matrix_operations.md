# Matrices in Context of Numpy

A matrix is a systematic arrangement of numbers (or elements) in rows and columns. An m × n matrix has `m` rows and `n` columns. The dimensions of the matrix are represented as m × n.

## Matrix Norm 

The matrix norm is a function that measures the size or magnitude of a matrix. Just as the absolute value of a number provides its magnitude, the norm gives the magnitude of a matrix. There are various ways to define matrix norms, with each having specific properties and applications.

### Vector Norm

A vector norm assigns a non-negative value to a vector in n-dimensional space, symbolizing its "length" or "magnitude". One common vector norm is the Euclidean norm, defined as:

$$
||x||_2 = \sqrt{\sum_{i=1}^n x_i^2}
$$

Given a matrix M, its Frobenius norm is:

$$
||\vec{M}||_F = \sqrt{\sum_i^m \sum_j^n (M_{ij})^2}
$$

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

Matrix multiplication, unlike element-wise multiplication, requires the number of columns of the first matrix to equal the number of rows of the second. The result is an m × p matrix. It is defined as:

$$ 
(M \times N)_{ij} = \sum_{k=1}^n M_{ik}N_{kj} 
$$

Example using NumPy:

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

The identity matrix, denoted as I, has ones on the diagonal and zeros everywhere else. If a matrix A has an inverse A^-1, then A×A−1=IA×A−1=I:

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

Matrix and vector arithmetic forms the basis of many computational problems. Both matrices and vectors can undergo a range of basic arithmetic operations. These operations can either act element-wise or involve more complex operations like matrix multiplication.

For two matrices or vectors of the same dimensions, element-wise operations are performed by combining corresponding elements.

- **Matrix Addition**: Each element in one matrix is added to its corresponding element in another matrix.
  
- **Scalar Multiplication**: Every element of the matrix or vector is multiplied by a scalar value.

Examples of element-wise operations in NumPy:

| Operation     | Function                  | Operator   |
|---------------|---------------------------|------------|
| addition      | `np.add(arr_1, arr_2)`    | `arr_1 + arr_2` |
| subtraction   | `np.subtract(arr_1, arr_2)`| `arr_1 - arr_2` |
| multiplication| `np.multiply(arr_1, arr_2)`| `arr_1 * arr_2` |
| division      | `np.divide(arr_1, arr_2)`  | `arr_1 / arr_2` |

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

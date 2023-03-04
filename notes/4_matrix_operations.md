
# About Matrices
An m × n matrix is a rectangular table of numbers consisting of m rows and n columns. The number of rows and columns in a matrix is referred to as the matrix's dimensions, and it is written as m × n. 

## Matrix Norm 

Matrix norm is a mathematical concept used to measure the magnitude of a matrix. It is a generalization of the concept of absolute value to matrices. The matrix norm can be defined in different ways, each of which has different properties and applications.

One of the most common ways to define the matrix norm is through the use of the vector norm. A vector norm is a function that assigns a non-negative value to a vector, which represents its "length" or "magnitude". For example, the Euclidean norm of a vector x in n-dimensional space is defined as:

$$||x||_2 = \sqrt{\sum_{i=1}^n x_i^2}$$

Given a matrix M, we can define its norm as:

$$||\vec{M}||p = \sqrt[p]{\sum_i^m \sum_j^n (M{ij})^p}$$

There are different ways to compute the norm of a matrix in NumPy. One way is to use the numpy.linalg.norm function, which can compute various types of norms, including the Frobenius norm and the spectral norm.

For example, to compute the Frobenius norm of a matrix A, you can use:

```Python
import numpy as np
A = np.array([[1, 2], [3, 4]])
norm_A = np.linalg.norm(A, 'fro')
print(norm_A)
```
Expected output:

```
5.477225575051661
```

To compute the spectral norm of a matrix A, you can use:

```Python
import numpy as np
A = np.array([[1, 2], [3, 4]])
norm_A = np.linalg.norm(A, 2)
print(norm_A)
```
Expected output:

```
5.464985704219043
```

Another important property of matrix norms is that they satisfy the following inequality:

$$||AB|| \leq ||A||\cdot||B||$$

This is known as the sub-multiplicative property of matrix norms. In other words, the norm of the product of two matrices is bounded by the product of their norms. This property is useful in many applications, such as bounding the error of numerical computations involving matrices.

Finally, it's worth noting that the norm of a matrix can be used to define a distance metric between matrices. For example, the distance between two matrices A and B can be defined as:

$$d(A,B) = ||A-B||$$

This distance metric satisfies the properties of a metric, such as non-negativity, symmetry, and the triangle inequality.

## Matrix multiplication

In order for M × N to be defined, the number of rows of N has to equal the number of columns of M. The product of an m × n matrix and an n × p matrix is an m × p matrix. Matrix multiplication is performed by multiplying the corresponding elements of the rows and columns, then summing the products of each pair. This process is repeated for each element in the resulting matrix.

$$M_{ij} = \sum_{k=1}^p P_{ik}Q_{kj}$$

Here's an example of matrix multiplication using NumPy:

```Python
import numpy as np
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

## Matrix transpose

The transpose of a matrix is a reversal of its rows with its columns. The transpose of an m × n matrix is an n × m matrix.

```Python
M = np.array([[-4, 5], [1, 7], [8, 3]])
print(np.transpose(M))
```

Expected output:

```
[[-4  1  8]
 [ 5  7  3]]
```

## Determinants

A matrix with the same number of elements in rows and columns is called a square matrix. Square matrices have determinants, which can be calculated using the determinant function from NumPy's linalg module.

```Python
M = np.array([[-4, 5], [1, 7]])
print(np.linalg.det(M))
```

Expected output:

```
-33
```

## Identity Matrix

The identity matrix is a square matrix with ones on the diagonal and zeros elsewhere. The inverse of a square matrix M is a matrix of the same size, N, such that $M \times N = I$.

```Python
M = np.array([[-4, 5], [1, 7]])
print(np.linalg.inv(M))
```

Expected output:

```
[[-0.21212121  0.15151515]
 [ 0.03030303  0.12121212]]
```

## Rank of a matrix

The rank of a matrix is the number of linearly independent rows or columns of the matrix. In other words, it is the maximum number of rows or columns that can be linearly combined to form a new row or column in the matrix. A matrix is said to be of full rank if its rank is equal to the minimum of the number of rows and columns.

In NumPy, we can compute the rank of a matrix using the np.linalg.matrix_rank function. Here's an example:

```Python
import numpy as np

# Create a matrix
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Compute the rank of the matrix
rank = np.linalg.matrix_rank(A)

print("Rank of A:", rank)
```

Output:

```
Rank of A: 2
```

In this example, the rank of matrix A is 2. This means that there are only 2 linearly independent rows or columns in the matrix.

We can also use the rank of a matrix to determine if the matrix is singular. A matrix is said to be singular if its rank is less than the number of columns (or rows). In other words, a matrix is singular if it does not have a unique inverse. Here's an example of how to check if a matrix is singular using its rank:

```Python
import numpy as np

# Create a singular matrix
A = np.array([[1, 2], [2, 4]])

# Compute the rank of the matrix
rank = np.linalg.matrix_rank(A)

if rank < A.shape[1]:
    print("Matrix A is singular.")
else:
    print("Matrix A is not singular.")
```

Output:

```
Matrix A is singular.
```

<h1>Summary of matrix and vector operations</h1>

Matrix addition and scalar multiplication work the same way as for vectors. In matrix addition, each element of one matrix is added to the corresponding element of another matrix. In scalar multiplication, each element of a matrix is multiplied by a scalar.

Element wise operations:

| Operation | Function | Operator |
| --- | --- | --- |
| addition |  np.add(arr_1, arr_2) | arr_1 + arr_2 |
| subtraction | np.subtract(arr_1, arr_2) | arr_1 - arr_2 |
| multiplication |  np.multiply(arr_1, arr_2) | arr_1 * arr_2 |
| division | np.divide(arr_1, arr_2) | arr_1 / arr_2 |

```Python
import numpy as np
arr_1 = np.array([1, 2, 3, 4])
arr_2 = np.array([1, 2, 3, 4])
print(arr_1 - arr_2)
```

Expected output:

```
[0 0 0 0]
```

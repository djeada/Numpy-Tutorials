
<h1>About matrices</h1>
An m×n matrix is a rectangular table of numbers consisting of m rows and n columns. 

Matrix norm is defined as:

$$||\vec{M}||_p = \sqrt[p]{\sum_i^m \sum_j^n (M_{ij})^p}$$

Matrix addition and scalar multiplication for matrices work the same way as for vectors. 

In order for  M⋅N to be defined, the number of rows of  N  has to equal the number of columns of M.
The product of an m × n matrix and an n × p matrix is an m × p matrix.

$$M_{ij} = \sum_{k=1}^p P_{ik}Q_{kj}$$


```Python
M = np.array([[-4, 5], [1, 7], [8, 3]])
N = np.array([[3, -5, 2, 7], [-5, 1, -4, -3]])
print(np.dot(M, N))
```

Expected output:

```
[[-37  25 -28 -43]
 [-32   2 -26 -14]
 [  9 -37   4  47]
```

The transpose of a matrix is a reversal of its rows with its columns.

```Python
M = np.array([[-4, 5], [1, 7], [8, 3]])
print(np.transpose(M))
```

Expected output:

```
[[-4  1  8]
 [ 5  7  3]]
```

A matrix with the same number of elements in rows and colums is called a <b>square matrix</b>. 
Square matrices have determinants.

```Python
M = np.array([[-4, 5], [1, 7]])
print(np.linalg.det(M))
```

Expected output:

```
-33
```

The <b>identity matrix</b> is a square matrix with ones on the diagonal and zeros elsewhere.
The <b>inverse</b> of a square matrix M is a matrix of the same size, N, such that M⋅N=I.

```Python
M = np.array([[-4, 5], [1, 7]])
print(np.linalg.inv(M))
```

Expected output:

```
[[-0.21212121  0.15151515]
 [ 0.03030303  0.12121212]]
```

The number of linearly independent columns or rows of a m x n matrix M is denoted by <b>rank (M)</b>.
An augmented matrix is a matrix M that has been concatenated with a vector v and written as [M,v]. “M augmented with v,” as it is usually written.

<h1>Summary of matrix and vector operations</h1>

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

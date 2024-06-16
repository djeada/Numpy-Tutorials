## Systems of Linear Equations

Systems of linear equations are a cornerstone of linear algebra and play a crucial role in various fields such as engineering, physics, computer science, and economics. These systems involve multiple linear equations that share common variables. By utilizing matrix notation, we can represent and solve these systems efficiently, leveraging the power of matrix operations.

### Matrix Representation

A system of linear equations can be succinctly expressed in matrix form. Consider a system of linear equations as follows:

\n$$
\begin{cases}
a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n = b_1 \\
a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n = b_2 \\
\vdots \\
a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n = b_m
\end{cases}
$$\n

This system can be represented in matrix form as:

\n$$
Mx = y
$$\n

where:
- $M$ is an $m \times n$ matrix containing the coefficients of the variables.
- $x$ is a column vector of $n$ variables.
- $y$ is a column vector of $m$ constants.

For example, the system:
\n$$
\begin{cases}
2x_1 + 0x_2 + 5x_3 = 10 \\
3x_1 + 4x_2 + 8x_3 = 15 \\
2x_1 + 7x_2 + 3x_3 = 5
\end{cases}
$$\n
can be written as:
\n$$
\begin{pmatrix}
2 & 0 & 5 \\
3 & 4 & 8 \\
2 & 7 & 3
\end{pmatrix}
\begin{pmatrix}
x_1 \\
x_2 \\
x_3
\end{pmatrix}
=
\begin{pmatrix}
10 \\
15 \\
5
\end{pmatrix}
$$\n

### Possible Solutions

For a given matrix equation $Mx = y$, there are three possible types of solutions:

I. **No Solution**

A system has no solution if the rank of the augmented matrix $[M|y]$ is greater than the rank of $M$:

\n$$
\text{rank}([M|y]) > \text{rank}(M)
$$\n

This situation occurs when the system is inconsistent, meaning the equations contradict each other.

II. **Unique Solution**

A system has a unique solution if the rank of $M$ equals the rank of the augmented matrix and this rank equals the number of variables $n$:

\n$$
\text{rank}([M|y]) = \text{rank}(M) = n
$$\n

In this case, the system is consistent and the equations are independent, leading to a single solution.

III. **Infinite Solutions**

A system has infinitely many solutions if the rank of both $M$ and the augmented matrix is the same, but this rank is less than the number of variables $n$:

\n$$
\text{rank}([M|y]) = \text{rank}(M) < n
$$\n

This occurs when the system has dependent equations, resulting in multiple solutions.

### Solving with NumPy

Python's NumPy library provides efficient tools for solving systems of linear equations. The `linalg.solve()` function is particularly useful for finding solutions to these systems. Given a matrix $M$ and a column vector $y$, it returns the vector $x$ that satisfies the equation $Mx = y$.

Here is an example of solving a system of linear equations using NumPy:

```python
import numpy as np

# Define the matrix M and vector y
M = np.array([[2, 0, 5], [3, 4, 8], [2, 7, 3]])
y = np.array([[10], [15], [5]])

# Solve for x
x = np.linalg.solve(M, y)
print(x)
```

Output:

```
[[ 0.65217391]
 [-0.2173913 ]
 [ 1.73913043]]
```

This indicates that the solutions to the system of equations are:

\n$$
x_1 = 0.65217391, \quad x_2 = -0.2173913, \quad x_3 = 1.73913043
$$\n

### Advanced Techniques for Solving Systems of Linear Equations with NumPy

While `numpy.linalg.solve()` is a straightforward and efficient method for solving systems of linear equations, there are several other techniques and functions in NumPy that can be employed depending on the nature of the system. These techniques include using matrix inverses, QR decomposition, and singular value decomposition (SVD). Additionally, handling special cases such as over-determined and under-determined systems requires specific methods.

#### Using the Matrix Inverse

For square matrices, another way to solve the system $Mx = y$ is by using the inverse of $M$. If $M$ is invertible, the solution can be found as:

\n$$
x = M^{-1}y
$$\n

This method, however, is computationally more expensive and less numerically stable than using `numpy.linalg.solve()`.

Example:

```python
import numpy as np

# Define the matrix M and vector y
M = np.array([[2, 0, 5], [3, 4, 8], [2, 7, 3]])
y = np.array([[10], [15], [5]])

# Calculate the inverse of M
M_inv = np.linalg.inv(M)

# Solve for x
x = np.dot(M_inv, y)
print(x)
```

Output:

```
[[ 0.65217391]
 [-0.2173913 ]
 [ 1.73913043]]
```

#### QR Decomposition

QR decomposition is useful for solving systems of linear equations, especially when dealing with over-determined systems (more equations than variables). QR decomposition decomposes matrix $M$ into an orthogonal matrix $Q$ and an upper triangular matrix $R$.

Example:

```python
import numpy as np

# Define the matrix M and vector y
M = np.array([[2, 0, 5], [3, 4, 8], [2, 7, 3]])
y = np.array([[10], [15], [5]])

# Perform QR decomposition
Q, R = np.linalg.qr(M)

# Solve for x
x = np.linalg.solve(R, np.dot(Q.T, y))
print(x)
```

Output:

```
[[ 0.65217391]
 [-0.2173913 ]
 [ 1.73913043]]
```

#### Singular Value Decomposition (SVD)

SVD is a powerful method that can handle both over-determined and under-determined systems. It decomposes matrix $M$ into three matrices $U$, $S$, and $V^T$ such that $M = U \Sigma V^T$.

Example:

```python
import numpy as np

# Define the matrix M and vector y
M = np.array([[2, 0, 5], [3, 4, 8], [2, 7, 3]])
y = np.array([[10], [15], [5]])

# Perform SVD
U, S, VT = np.linalg.svd(M)

# Compute the pseudo-inverse of S
S_inv = np.diag(1/S)

# Solve for x using the pseudo-inverse
x = np.dot(VT.T, np.dot(S_inv, np.dot(U.T, y)))
print(x)
```

Output:

```
[[ 0.65217391]
 [-0.2173913 ]
 [ 1.73913043]]
```

#### Handling Over-Determined and Under-Determined Systems

For over-determined systems (more equations than variables), we often seek a least-squares solution. NumPy provides `numpy.linalg.lstsq()` for this purpose.

Example of an over-determined system:

```python
import numpy as np

# Define the over-determined matrix M and vector y
M = np.array([[1, 1], [1, -1], [1, 0]])
y = np.array([2, 0, 1])

# Solve using least squares
x, residuals, rank, s = np.linalg.lstsq(M, y, rcond=None)
print(x)
```

Output:

```
[1. 1.]
```

For under-determined systems (fewer equations than variables), we often seek a solution with the minimum norm. This can be achieved using the pseudo-inverse of the matrix, which can be computed using `numpy.linalg.pinv()`.

Example of an under-determined system:

```python
import numpy as np

# Define the under-determined matrix M and vector y
M = np.array([[1, 2, 3], [4, 5, 6]])
y = np.array([7, 8])

# Compute the pseudo-inverse of M
M_pinv = np.linalg.pinv(M)

# Solve for x
x = np.dot(M_pinv, y)
print(x)
```

Output:

```
[-6.94444444  0.22222222  7.38888889]
```

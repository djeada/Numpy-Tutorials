## Systems of Linear Equations
A couple of linear equations with the same variables is known as a system of linear equations. In linear algebra, we can represent a system of linear equations in matrix form as $Mx = y$, where M is an m × n matrix, x is a column vector of n variables, and y is a column vector of m constants.

If a system of linear equations is given in a matrix form, $Mx = y$, where N is an m×n matrix, then there are three possibilities:

1. There is no solution for x.

In this case, the rank of the augmented matrix $[M, y]$ is greater than the rank of $M$. This can be represented as:

$$rank([M,y]) > rank(M)$$

2. There is a unique solution for x.

In this case, the rank of the augmented matrix $[M, y]$ is the same as the rank of $M$. This can be represented as:

$$rank([M,y]) = rank(M)$$

3. There is an infinite number of solutions for x.

In this case, the rank of the augmented matrix $[M, y]$ is the same as the rank of $M$, but the rank of M is less than the number of variables n. This can be represented as:

$$rank([M,y]) = rank(M) < n$$

In NumPy, we can use the linalg.solve() function to solve systems of linear equations. The function takes two arguments: the matrix M and the column vector y. Here is an example:

```Python
import numpy as np

matrix = np.matrix([[2, 0, 5], [3, 4, 8], [2, 7, 3]])
y = np.matrix([[10], [15], [5]])

x = np.linalg.solve(matrix, y)
print(x)
```

The output will be:

```
[[ 0.65217391]
 [-0.2173913 ]
 [ 1.73913043]]
```

This means that the solution to the system of linear equations is:

$x_1 = 0.65217391$
$x_2 = -0.2173913$
$x_3 = 1.73913043$

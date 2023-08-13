## Systems of Linear Equations

Systems of linear equations are foundational in linear algebra. They involve multiple linear equations that share common variables. This system can be represented in matrix notation, making matrix operations pivotal for their solutions.

### Matrix Representation

A system of linear equations can be represented in matrix form as:

$$
Mx = y 
$$

Here:
- $M$ is an $m \times n$ matrix containing the coefficients of the variables.
- $x$ is a column vector of $n$ variables.
- $y$ is a column vector of $m$ constants.

### Possible Solutions

For a given matrix equation $Mx = y$ :

1. **No Solution**: 
 - This situation arises when the rank of the augmented matrix $[M|y]$ exceeds the rank of $M$ .

$$
rank([M|y]) > rank(M) 
$$

 - This means that the system is inconsistent and does not have a solution.

2. **Unique Solution**:
 - The system has a single solution if the rank of $M$ is equal to the rank of the augmented matrix and also equals the number of variables $n$ .

$$
rank([M|y]) = rank(M) = n 
$$

3. **Infinite Solutions**:
 - Here, the rank of both $M$ and the augmented matrix is the same, but less than $n$, the number of variables.

$$
rank([M|y]) = rank(M) < n 
$$

 - The system has multiple solutions due to the redundant equations.

### Solving with NumPy

NumPy's `linalg.solve()` function provides an easy way to solve these systems. Given a matrix $M$ and column vector $y$ , it returns the vector $x$ that satisfies the equation.

```Python
import numpy as np

M = np.array([[2, 0, 5], [3, 4, 8], [2, 7, 3]])
y = np.array([[10], [15], [5]])

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

$x_1=0.65217391x1​=0.65217391$

$x_2=−0.2173913x2​=−0.2173913$

$x_3=1.73913043x3​=1.73913043$

Understanding systems of linear equations and their matrix representations paves the way for various applications in science, engineering, and data science. 

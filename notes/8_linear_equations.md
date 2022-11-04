<h1>Systems of linear equations</h1>
A couple of linear equations with the same variables is known as a system of linear equations.

If a system of linear equations in given in a matrix form, Mx=y, where N is an m×n matrix, then there are three possibilites:

1. There is no solution for x.

rank([M,y]) = rank(M) + 1

2. There is a unique solution for x.

rank([M,y]) = rank(M)

3. There is an infinite number of solutions for x

rank([M,y]) = rank(M) and rank(M) < n

```Python
matrix = np.matrix([[2, 0, 5], [3, 4, 8], [2, 7, 3]])
y = np.matrix([[10], [15], [5]])
print(np.solve(matrix, y))
```

Expected output:

```
[[ 0.65217391]
 [-0.2173913 ]
 [ 1.73913043]]
```

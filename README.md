# NumPy
NumPy tutorials.

TODO: 
* extracting submatrices, rows and cols
* replace i'th row or col
* generally transformations between diffrent shapes
* ravel, flatten

Table of Contents
=================

<!--ts-->
   * [About NumPy](#About-NumPy)
   * [Creating an array](#Creating-an-array)
   * [Joining and splitting arrays](#Joining-and-splitting-arrays)
   * [Accessing elements](#Accessing-elements)
   * [About vectors](#About-vectors)
   * [About matrices](#About-matrices)
   * [Summary of matrix and vector operations](#Summary-of-matrix-and-vector-operations)
   * [Matrix manipulations](#Matrix-manipulations)
   * [Random numbers](#Random-numbers)
   * [Numpy statistics](#Numpy-statistics)
   * [Code samples](#Code-samples)

<h1>About NumPy</h1>
Numpy stands for Numerical Python.

* It's a Python library for manipulating arrays.
* It also includes functions for working in the linear algebra domain, the fourier transform, and matrices.
* It's a free and open source initiative.
* It is very effective for scientific computations.

I highly recommend to read this <a href="https://betterprogramming.pub/numpy-illustrated-the-visual-guide-to-numpy-3b1d4976de1d">article</a>. 

<h1>Creating an array</h1>

<h2>Creating from list</h2>

```Python
import numpy as np
arr = np.array([1, 2, 3, 4])
print(arr)
print(type(arr))
```

Expected output:

```
[1 2 3 4]
<class 'numpy.ndarrray'>
```

Arrays are objects of <i>Ndarray</i> class. It provides a lot of useful functions for working with arrays.

<h2>Evenly spaced numbers</h2>

The np.linspace(start, end, n) function return evenly spaced n numbers over a specified interval.

```Python
import numpy as np
arr = np.linspace(1, 5, 9)
print(arr)
```

Expected output:

```
[1. 1.5 2. 2.5 3. 3.5 4. 4.5 5.]
```

<h1>Joining and splitting arrays</h1>

* Stacking is the process of joining a sequence of identical-dimension arrays around a new axis.
The axis parameter determines the position of the new axis in the result's dimensions.

* Concatenating refers to joining a sequence of arrays along an existing axis. 
* Appending means adding values along the specified axis at the end of the array.
* Spliting is the process of breaking an array into sub-arrays of identical size.

<h1>Accessing elements</h1>

NumPy arrays have indices that begin with 0. The first element has an index of 0, the second element has an index of 1, and so on.

```Python
import numpy as np
arr = np.array([1, 2, 3, 4])
print(arr[1])
```

Expected output:

```
2
```

In matrices you have to first provide row index and then column index.

![Matrix](https://github.com/djeada/Numpy/blob/main/resources/matrix.png)


```Python
import numpy as np
arr = np.array([
  [7, 1, 2, 6], 
  [6, 4, 9, 3], 
  [2, 1, 4, 5], 
  [2, 7, 3, 8]
])
print(arr[1][2])
print(arr[3][0])
```

Expected output:

```
9
2
```

Numpy arrays are mutable. You can change the value under index.

```Python
import numpy as np
arr = np.array([1, 2, 3, 4])
arr[2] = 5
print(arr)
```

Expected output:

```
[1 2 5 4]
```

You can access group of elements with slicing.
You pass slice instead of single index to square brackets. <i>\[start\:end\:step\] </i>

* If you don't specify a value for start, it will default to 0.
* If you don't specify a value for end, it will default to array's size.
* If you don't specify a value for step, it will default to 1.

```Python
import numpy as np
arr = np.array([1, 2, 3, 4])
print(arr[::2])
print(arr[1:])
print(arr[:-3])
```

Expected output:

```
[1 3]
[2 3 4]
[1]
```

<h1>About vectors</h1>

A vector in R^n is an n-tuple. 

In a <b>row vector</b>, the elements of the vector are written next to each other, and in a <b>column vector</b>, the elements of the vector are written on top of each other.

A column <b>vector's transpose</b> is a row vector of the same length, and a row vector's transpose is a column vector.

The <b>norm</b> is a way to measure vector's length. Depending on the metric used, there are a variety of ways to define the length of a vector . The most common is L2 norm, which uses the distance formula.

![vector norm](https://github.com/djeada/Numpy/blob/main/resources/vector_norm.png)


Operations:

1) Vector addition: 
The pairwise addition of respective elements.

```Python
arr_1 = np.array([9, 2, 5])
arr_2 = np.array([-3, 8, 2])
print(np.add(arr_1, arr_2))
```

Expected output:

```
[ 6 10  7]
```

2) Scalar multiplication:
The product of each element of the vector by the given scalar.

```Python
arr = np.array([6, 3, 4])
scalar = 2
print(scalar * arr)
```

Expected output:

```
[12  6  8]
```

3) Dot product:
The sum of pairwise products of respective elements.

![dot product](https://github.com/djeada/Numpy/blob/main/resources/dot_product.png)


```Python
arr_1 = np.array([9, 2, 5])
arr_2 = np.array([-3, 8, 2])
print(np.dot(arr_1, arr_2))
```

Expected output:

```
-1
```

4) The cross product:

  The cross product's geometric representation is a vector perpendicular to both v and w, with a length equal to the region enclosed by the parallelogram formed by the two vectors.

![cross product](https://github.com/djeada/Numpy/blob/main/resources/cross_product.png)

  Where <i>n</i> is a unit vector perpendicular to plane.

```Python
arr_1 = np.array([9, 2, 5])
arr_2 = np.array([-3, 8, 2])
print(np.cross(arr_1, arr_2))
```

Expected output:

```
[[-36 -33  78]]
```

5) The angle between two vectors:

![dot product](https://github.com/djeada/Numpy/blob/main/resources/dot_product.png)

If the angle between the vectors θ=π/2, then the vectors are said to be perpendicular or orthogonal, and the dot product is 0.

```Python
arr_1 = np.array([9, 2, 5])
arr_2 = np.array([-3, 8, 2])
print(np.arccos(np.dot(arr_1, arr_2)/(np.norm(arr_1)*np.norm(arr_2)))
```

Expected output:

```
1.582
```

<h1>About matrices</h1>
An m×n matrix is a rectangular table of numbers consisting of m rows and n columns. 

Matrix norm is defined as:

![matrix norm](https://github.com/djeada/Numpy/blob/main/resources/matrix_norm.png)


Matrix addition and scalar multiplication for matrices work the same way as for vectors. 

In order for  M⋅N to be defined, the number of rows of  N  has to equal the number of columns of M.
The product of an m × n matrix and an n × p matrix is an m × p matrix.

![matrix mult](https://github.com/djeada/Numpy/blob/main/resources/matrix_mult.png)


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

<h1>Matrix manipulations</h1>

The term "reshape" refers to changing the shape of an array.
The number of elements in each dimension determines the shape of an array.
We may adjust the number of elements in each dimension or add or subtract dimensions.

```Python
import numpy as np
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(arr.reshape(2,5))
```

Expected output:

```
[[1 2 3 4 5]
[6 7 8 9 10]]
```

Flatten returns a one-dimensional version of the array.

```Python
import numpy as np
arr = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10]
])
print(arr.flatten())
```

Expected output:

```
[1 2 3 4 5 6 7 8 9 10]
```

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

<h1>Random numbers</h1>

1. Floats between 0 and 1.

```Python
np.random.rand(d0, d1...)
```
  It generate an array with random numbers (float) that are uniformly distributed between 0 and 1.
  The parameter allows you to specify the shape of the array.
  
2. Standard normal distribution.

```Python
np.random.randn(d0, d1...)
```
  It generates an array with random numbers (float) that are normally distrbuted. Mean = 0, Stdev (standard deviation) = 1.
  
3. Random integers within range

```Python
np.random.randint(low, high=None, size=None)
```

It generates an array with random numbers (integers) that are uniformly distributed between 0 and given number.

4. Random floats within range

```Python
np.random.uniform(low=0.0, high=1.0, size=None)
```

It generates an array with random numbers (float) between given numbers.

<h1>Numpy statistics</h1>

Statistics is a field of study that uses data to make observations about populations (groups of objects). In statistics textbooks they are often called "distributions" instead of "populations". Probability is integral part of statistics.

Basic statistical operations include:

1. Mean

![mean](https://github.com/djeada/Numpy/blob/main/resources/mean.png)

2. Median

![median](https://github.com/djeada/Numpy/blob/main/resources/median.png)

3. Variance

![variance](https://github.com/djeada/Numpy/blob/main/resources/variance.png)

4. Standard deviation

![standard deviation](https://github.com/djeada/Numpy/blob/main/resources/std.png)

| Operation | Function |
| --- | --- |
| mean |  np.mean(arr) |
| median | np.median(arr) | 
| variance |  np.var(arr) |
| standard deviation | np.std(arr) | |

<h1>Code samples</h1>

* <a href="https://github.com/djeada/Numpy/blob/main/src/1_creating_arrays.py">Creating arrays.</a>
* <a href="https://github.com/djeada/Numpy/blob/main/src/2_join_split.py">Joining and splitting.</a>
* <a href="https://github.com/djeada/Numpy/blob/main/src/3_accessing_modifying_elements.py">Accessing and modyfing elements.</a>
* <a href="https://github.com/djeada/Numpy/blob/main/src/4_searching.py">Searching.</a>
* <a href="https://github.com/djeada/Numpy/blob/main/src/5_vector_operations.py">Vector operations.</a>
* <a href="https://github.com/djeada/Numpy/blob/main/src/6_matrix_operations.py">Matrix operations.</a>
* <a href="https://github.com/djeada/Numpy/blob/main/src/7_manipulating_matrices.py">Matrix manipulations.</a>
* <a href="https://github.com/djeada/Numpy/blob/main/src/8_linear_equations.py">Linear equations</a>
* <a href="https://github.com/djeada/Numpy/blob/main/src/9_statistics_and_random_numbers.py">Statistics and random numbers</a>

# Numpy
Tutorials about Numpy.

Table of Contents
=================

<!--ts-->
   * [About Numpy](#About-Numpy)
   * [Creating an array](#Creating-an-array)
   * [Joining and splitting arrays](#Joining-and-splitting-arrays)
   * [Accessing elements](#Accessing-elements)
   * [Matrix and vector operations](#Matrix-and-vector-operations)
   * [Matrix manipulations](#Matrix-manipulations)
   * [Random numbers](#Random-numbers)
   * [Numpy Statistics](#Numpy-Statistics)
   * [Code Samples](#Code-Samples)

<h1>About Numpy</h1>
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

<h1>Accessing elements </h1>

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
You pass slice instead of single index to square brackets. <i>\[start:end:step\] </i>

* If you don't pass start it will keep it's default value 0.
* If you don't pass end it will keep it's default value size().
* If you don't pass step it will keep it's deafult value 1.

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
<h1>Matrix and vector operations</h1>

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


<h1>Random Numbers</h1>

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

<h1>Numpy Statistics</h1>

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

<h1>Code Samples</h1>

* </a href="https://github.com/djeada/Numpy/blob/main/src/1_creating_arrays.py">Creating arrays.</a>
* </a href="https://github.com/djeada/Numpy/blob/main/src/2_join_split.py">Joining and splitting.</a>
* </a href="https://github.com/djeada/Numpy/blob/main/src/3_accessing_modifying_elements.py">Accessing and modyfing elements.</a>
* </a href="https://github.com/djeada/Numpy/blob/main/src/4_searching.py">Searching.</a>
* </a href="https://github.com/djeada/Numpy/blob/main/src/5_matrix_operations.py">Matrix operations.</a>
* </a href="https://github.com/djeada/Numpy/blob/main/src/6_manipulating_matrices.py">Matrix manipulations.</a>
* </a href="https://github.com/djeada/Numpy/blob/main/src/7_linear_equations.py">Linear equations</a>
* </a href="https://github.com/djeada/Numpy/blob/main/src/8_statistics_and_random_numbers.py">Statistics and random numbers</a>

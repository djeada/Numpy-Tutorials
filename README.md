# Numpy
Tutorials about Numpy.

<h1>About Numpy</h1>
Numpy stands for Numerical Python.

It is a Python library used for working with arrays.

It also has functions for working in domain of linear algerba, fourer transform and matrices.

It is an open source project.

Efficient for scientific computation.

I highly recommend to read this <a href="https://betterprogramming.pub/numpy-illustrated-the-visual-guide-to-numpy-3b1d4976de1d">article</a>. 

<h1>Creating an array</h1>

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

Ndarray provides a lot of useful functions.

<h1>Accessing elements </h1>

The indices in NumPY arrays start with 0. The first element has index 0, the second index 1 and so on.

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



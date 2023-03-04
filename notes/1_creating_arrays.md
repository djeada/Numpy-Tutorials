# Creating an array

Numpy provides a high-performance multi-dimensional array object called `ndarray`. In this note, we will discuss how to create numpy arrays.

## Creating from a List

To create an array from a list, we use the `np.array()` function. Here is an example:

```Python
import numpy as np
arr = np.array([1, 2, 3, 4])
print(arr)
print(type(arr))
```

Expected output:

```
[1 2 3 4]
<class 'numpy.ndarray'>
```

Arrays are objects of the ndarray class. It provides a lot of useful functions for working with arrays.

## Creating from a Tuple

We can also create arrays from tuples. The process is the same as creating arrays from lists. Here is an example:

```Python
import numpy as np
arr = np.array((1, 2, 3, 4))
print(arr)
print(type(arr))
```

Expected output:

```
[1 2 3 4]
<class 'numpy.ndarray'>
```

## Creating an Array of Zeros

We can create an array of zeros using the `np.zeros()` function. Here is an example:

```Python
import numpy as np
arr = np.zeros((2, 3))
print(arr)
```

Expected output:

```
[[0. 0. 0.]
 [0. 0. 0.]]
```

## Creating an Array of Ones

We can create an array of ones using the `np.ones()` function. Here is an example:

```Python
import numpy as np
arr = np.ones((2, 3))
print(arr)
```

Expected output:

```
[[1. 1. 1.]
 [1. 1. 1.]]
```

## Creating an Array of Random Numbers

We can create an array of random numbers using the `np.random.rand()` function. Here is an example:

```Python
import numpy as np
arr = np.random.rand(2, 3)
print(arr)
```

Output:

```
[[0.72599639 0.09498103 0.26297852]
 [0.24481388 0.18111837 0.94409848]]
```

## Creating an Array with Evenly Spaced Numbers

We can create an array with evenly spaced numbers using the `np.linspace()` function. Here is an example:

```Python
import numpy as np
arr = np.linspace(1, 5, 9)
print(arr)
```

Output:

```
[1.  1.5 2.  2.5 3.  3.5 4.  4.5 5. ]
```

The `np.linspace()` function returns evenly spaced n numbers over a specified interval.

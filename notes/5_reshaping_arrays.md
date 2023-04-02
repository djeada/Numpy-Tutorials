## Changing shape of a matrix

The term "reshape" refers to changing the shape of an array. The number of elements in each dimension determines the shape of an array. We may adjust the number of elements in each dimension or add or subtract dimensions.

To reshape an array, we use the reshape() method of NumPy arrays. This method returns a new array with the same data but a different shape.

```Python
import numpy as np
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(arr.reshape(2, 5))
```

Expected output:

```
[[1 2 3 4 5]
 [6 7 8 9 10]]
```

In this example, we take the one-dimensional array arr and reshape it into a two-dimensional array with two rows and five columns. Note that the new shape should be compatible with the original shape, which means that the total number of elements must remain the same.

We can also use the reshape() method to change the number of dimensions of an array. For example, we can convert a one-dimensional array into a two-dimensional row or column vector:

```Python
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6])

# Convert to row vector
row_vector = arr.reshape(1, 6)
print(row_vector)

# Convert to column vector
column_vector = arr.reshape(6, 1)
print(column_vector)
```

Expected output:

```
[[1 2 3 4 5 6]]
[[1]
 [2]
 [3]
 [4]
 [5]
 [6]]
```

The flatten() method returns a one-dimensional version of the array. It is equivalent to reshaping the array into a one-dimensional array with the reshape(-1) method.

```Python

import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])

# Flatten the array
flat_arr = arr.flatten()
print(flat_arr)

# Reshape the array into a one-dimensional array
one_d_arr = arr.reshape(-1)
print(one_d_arr)
```

Expected output:

```
[1 2 3 4 5 6]
[1 2 3 4 5 6]
```

Note that flattening an array creates a copy of the original array, so any changes made to the flattened array will not affect the original array.

It's worth noting that reshaping an array can sometimes cause an error if the new shape is not compatible with the original shape. In this case, NumPy will raise a ValueError with a message indicating the problem.

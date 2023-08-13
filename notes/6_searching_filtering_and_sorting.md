# Searching, Filtering, and Sorting with NumPy

NumPy offers a suite of functions designed for searching within, filtering, and sorting arrays. These capabilities are indispensable when managing and preprocessing datasets, particularly large ones.

## Searching

To locate the indices of specific values or those that satisfy a particular condition within an array, you can utilize `np.where()`.

### Example with 1D Array:

```Python
import numpy as np

array = np.array([0, 1, 2, 3, 4, 5])
indices = np.where(array == 2)
print(indices[0]) # Expected: [2]

selected_values = array[np.where((array > 1) & (array < 4))]
for value in selected_values:
    print(value) # Expected: 2, 3
```

### Example with 2D Array:

```Python
array_2D = np.array([[0, 1], [1, 1], [5, 9]])
indices = np.where(array_2D == 1)

for row, col in zip(indices[0], indices[1]):
    print(array_2D[row, col]) # Expected: 1, 1, 1
```

## Filtering

Boolean indexing enables the extraction of elements that satisfy specific conditions from an array.

Example:

```Python
array = np.array([0, 1, 2, 3, 4, 5])
filtered_array = array[(array > 1) & (array < 4)]
print(filtered_array) # Expected: [2, 3]
```

## Sorting

For sorting arrays, NumPy offers the np.sort() function. It produces a sorted copy of the array while leaving the initial array untouched.

### Example with 1D Array:

```Python
array = np.array([3, 1, 4, 2, 5])
sorted_array = np.sort(array)
print(sorted_array) # Expected: [1, 2, 3, 4, 5]
```

### Example with 2D Array:

When sorting multidimensional arrays, you can specify the sorting axis using the axis parameter.

```Python
array_2D = np.array([[3, 1], [4, 2], [5, 0]])
sorted_array_2D = np.sort(array_2D, axis=0)
print(sorted_array_2D)
```

Expected output:

```
[[3, 0],
 [4, 1],
 [5, 2]]
```

These functions significantly streamline tasks associated with searching, filtering, and sorting, making data manipulation in NumPy both efficient and intuitive.

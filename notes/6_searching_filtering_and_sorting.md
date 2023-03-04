# Searching, Filtering, and Sorting</h1>

NumPy provides several functions to search, filter and sort arrays. These functions allow for quick and efficient manipulation of data within arrays.

## Searching

The `np.where()` function can be used to find the index of a given value or the indices of all values meeting a given condition in an array. For example:

```Python
import numpy as np

array = np.array([0, 1, 2, 3, 4, 5])
number = 2
result = np.where(array == 2)
print(result[0]) # prints [2]

for i in np.where((array > 1) & (array < 4))[0]:
    num = array[i]
    print(num) # prints 2, 3
```

This function can also be used for 2D arrays:

```Python
array_2D = np.array([[0, 1], [1, 1], [5, 9]])
number = 1
result = np.where(array_2D == number)

for cor in list(zip(result[0], result[1])):
    num = array_2D[cor[0]][cor[1]]
    print(num) # prints 1, 1
```

## Filtering

Filtering can be done using boolean indexing. The following example filters the values between 1 and 4 from an array:

```Python
array = np.array([0, 1, 2, 3, 4, 5])
filtered_array = array[(array > 1) & (array < 4)]
print(filtered_array) # prints [2, 3]
```

## Sorting

Sorting arrays can be done using the `np.sort()` function. This function returns a sorted copy of the array, leaving the original array unchanged. For example:

```Python
array = np.array([3, 1, 4, 2, 5])
sorted_array = np.sort(array)
print(sorted_array) # prints [1, 2, 3, 4, 5]
```

You can also sort 2D arrays by specifying the axis parameter:

```Python
array_2D = np.array([[3, 1], [4, 2], [5, 0]])
sorted_array_2D = np.sort(array_2D, axis=0)
print(sorted_array_2D)
# prints [[3, 0],
#         [4, 1],
#         [5, 2]]
```

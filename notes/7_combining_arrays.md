# Joining and Splitting Arrays with Numpy

In NumPy, manipulating the structure of arrays is a common operation. Whether combining multiple arrays into one or splitting a single array into several parts, NumPy provides a set of intuitive functions to achieve these tasks efficiently.

## Stacking Arrays

Stacking is the technique of joining arrays along a new axis. The function for this is `np.stack()`, and the axis on which you want to stack the arrays can be specified using the `axis` parameter.

```Python
import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# Vertical stacking
c = np.stack((a, b))
print("Vertically stacked:
", c)

# Horizontal stacking
d = np.stack((a, b), axis=1)
print("
Horizontally stacked:
", d)
```

Expected output:

```
Vertically stacked:
[[[1 2]
  [3 4]]

 [[5 6]
  [7 8]]]

Horizontally stacked:
[[[1 2]
  [5 6]]

 [[3 4]
  [7 8]]]
```

## Concatenating Arrays

Concatenation merges arrays along an existing axis. Use `np.concatenate()` for this.

```Python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# Vertical concatenation
c = np.concatenate((a, b))
print("Vertically concatenated:
", c)

# Horizontal concatenation
d = np.concatenate((a, b), axis=1)
print("
Horizontally concatenated:
", d)
```

Expected output:

```
Vertically concatenated:
[[1 2]
 [3 4]
 [5 6]
 [7 8]]

Horizontally concatenated:
[[1 2 5 6]
 [3 4 7 8]]
```

## Appending to Arrays

You can append values to an array using `np.append()`. It adds values at the end along the specified axis.

```Python

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Append b to a
c = np.append(a, b)
print("Appended array:
", c)
```

Expected output:

```
Appended array:
[1 2 3 4 5 6]
```

## Splitting Arrays

Splitting breaks down an array into smaller chunks. `np.split()` is the function used to divide arrays.

### Regular and Custom Splits

```Python
a = np.array([1, 2, 3, 4, 5, 6])

# Split into three equal parts
b = np.split(a, 3)
print("Regular split:
", b)

# Split at the 2nd and 4th indices
c = np.split(a, [2, 4])
print("
Custom split:
", c)
```

Expected output:

```
Regular split:
[array([1, 2]), array([3, 4]), array([5, 6])]

Custom split:
[array([1, 2]), array([3, 4]), array([5, 6])]
```

Whether you're restructuring datasets or simply organizing data, understanding how to effectively join and split arrays in NumPy is crucial. These operations offer a solid foundation for more advanced array manipulations and data transformations.

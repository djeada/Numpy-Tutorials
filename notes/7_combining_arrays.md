# Joining and splitting arrays

There are several ways to combine or split arrays in NumPy.

## Stacking

Stacking is the process of joining a sequence of identical-dimension arrays around a new axis. The np.stack() function is used for this purpose. The axis parameter determines the position of the new axis in the result's dimensions.

```Python
import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# Stack arrays vertically (along rows)
c = np.stack((a, b))
print(c)

# Stack arrays horizontally (along columns)
d = np.stack((a, b), axis=1)
print(d)
```

Expected output:

```
# Vertically stacked arrays
[[[1 2]
  [3 4]]

 [[5 6]
  [7 8]]]

# Horizontally stacked arrays
[[[1 2]
  [5 6]]

 [[3 4]
  [7 8]]]
```

## Concatenating

Concatenating refers to joining a sequence of arrays along an existing axis. The np.concatenate() function is used for this purpose.

```Python
import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# Concatenate arrays vertically (along rows)
c = np.concatenate((a, b))
print(c)

# Concatenate arrays horizontally (along columns)
d = np.concatenate((a, b), axis=1)
print(d)
```

Expected output:

```
# Vertically concatenated arrays
[[1 2]
 [3 4]
 [5 6]
 [7 8]]

# Horizontally concatenated arrays
[[1 2 5 6]
 [3 4 7 8]]
```

## Appending

Appending means adding values along the specified axis at the end of the array. The np.append() function is used for this purpose.

```Python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Append values to end of array
c = np.append(a, b)
print(c)
```

Expected output:

```
[1 2 3 4 5 6]
```

## Splitting

Splitting is the process of breaking an array into sub-arrays of identical size. The np.split() function is used for this purpose.

```Python
import numpy as np

a = np.array([1, 2, 3, 4, 5, 6])

# Split array into three sub-arrays
b = np.split(a, 3)
print(b)

# Split array at specified positions
c = np.split(a, [2, 4])
print(c)
```

Expected output:

```
# Three sub-arrays
[array([1, 2]), array([3, 4]), array([5, 6])]

# Split at specified positions
[array([1, 2]), array([3, 4]), array([5, 6])]
```

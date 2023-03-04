## Accessing and Modifying Elements

Arrays have indices that start at 0, with the first element having an index of 0, the second element an index of 1, and so on. The elements can be accessed and modified using indexing and slicing.

## Accessing Elements in 1-D Arrays

To access a specific element in a 1-dimensional NumPy array, you can use its index inside square brackets.

```Python
import numpy as np
arr = np.array([1, 2, 3, 4])
print(arr[1])
```

Expected output:

```
2
```

## Accessing Elements in 2-D Arrays

In a 2-dimensional array, you have to provide the row index and then the column index to access a specific element. For example, consider the following matrix:

$$
\begin{bmatrix}
7 \&  1 \& 2 \&  6 \\
6 \&  4 \& 9 \&  3 \\
2 \&  1 \& 4 \&  5 \\
2 \&  7 \& 3 \&  8 \\
\end{bmatrix}
$$

To access the element at row 2 and column 3, you can do:

```Python
import numpy as np
arr = np.array([
  [7, 1, 2, 6], 
  [6, 4, 9, 3], 
  [2, 1, 4, 5], 
  [2, 7, 3, 8]
])
print(arr[1][2])
```

Expected output:

```
9
```

## Modifying Elements

Arrays are mutable, meaning that their elements can be modified after creation. You can modify an element in an array by assigning a new value to it using its index.

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

## Slicing 1-D Arrays

Slicing is a way to access a group of elements within a NumPy array. It returns a new array that contains the selected elements. To slice a NumPy array, you pass a slice object inside square brackets.

For a 1-dimensional NumPy array, you can slice it using the syntax `arr[start:end:step]`.

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

## Slicing 2-D Arrays

When slicing a 2D array, the first parameter refers to the row, while the second parameter refers to the column.

```Python
import numpy as np
arr = np.array([
  [7, 1, 2, 6], 
  [6, 4, 9, 3], 
  [2, 1, 4, 5], 
  [2, 7, 3, 8]
])
print(arr[0:2, 1:3])
```

Expected output:

```
[[1 2]
 [4 9]]
```

The first slice (0:2) extracts the first two rows, while the second slice (1:3) extracts columns 1 and 2. Therefore, the output consists of the elements in rows 0 and 1 and columns 1 and 2.

If you don't specify the starting index, it will start from the first row or column, and if you don't specify the ending index, it will go up to the last row or column.

```Python
import numpy as np
arr = np.array([
  [7, 1, 2, 6], 
  [6, 4, 9, 3], 
  [2, 1, 4, 5], 
  [2, 7, 3, 8]
])
print(arr[:3, 2:])
```

Expected output:

```
[[2 6]
 [9 3]
 [4 5]]
```

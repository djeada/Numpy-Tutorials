
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

$$
\begin{bmatrix}
7 \&  1 \& 2 \&  6 \\
6 \&  4 \& 9 \&  3 \\
2 \&  1 \& 4 \&  5 \\
2 \&  7 \& 3 \&  8 \\
\end{bmatrix}
$$

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


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


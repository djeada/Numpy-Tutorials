<h1>Creating an array</h1>

<h2>Creating from list</h2>

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

Arrays are objects of <i>Ndarray</i> class. It provides a lot of useful functions for working with arrays.

<h2>Evenly spaced numbers</h2>

The np.linspace(start, end, n) function return evenly spaced n numbers over a specified interval.

```Python
import numpy as np
arr = np.linspace(1, 5, 9)
print(arr)
```

Expected output:

```
[1. 1.5 2. 2.5 3. 3.5 4. 4.5 5.]
```

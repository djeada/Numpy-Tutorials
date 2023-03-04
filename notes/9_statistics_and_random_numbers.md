# Random numbers

Numpy provides several functions to generate random numbers with different distributions.

## Floats between 0 and 1

```Python
np.random.rand(d0, d1...)
```

This function generates an array with random numbers (float) that are uniformly distributed between 0 and 1. The parameter allows you to specify the shape of the array.

Example:

```Python
import numpy as np

# Generate an array of shape (2, 3) with random numbers between 0 and 1
rand_array = np.random.rand(2, 3)
print(rand_array)
```

Output:

```
[[0.34288804 0.14940791 0.64889261]
 [0.66280316 0.63574457 0.15233215]]
```

## Standard normal distribution

```Python
np.random.randn(d0, d1...)
```

This function generates an array with random numbers (float) that are normally distributed. Mean = 0, Stdev (standard deviation) = 1.

Example:

```Python
import numpy as np

# Generate an array of shape (2, 3) with random numbers from a normal distribution
rand_array = np.random.randn(2, 3)
print(rand_array)
```

Output:

```
[[ 0.37816256  1.22181034 -1.00768474]
 [ 1.49015915  0.60481905 -0.60007711]]
```

## Random integers within range

```Python
np.random.randint(low, high=None, size=None)
```

This function generates an array with random numbers (integers) that are uniformly distributed between the given low and high values.

Example:

```Python
import numpy as np

# Generate an array with 5 random integers between 0 and 9
rand_integers = np.random.randint(0, 10, size=5)
print(rand_integers)
```
Output:

```
[4 0 9 7 3]
```

## Random floats within range

```Python
np.random.uniform(low=0.0, high=1.0, size=None)
```

This function generates an array with random numbers (float) between the given low and high values.

Example:

```Python

import numpy as np

# Generate an array of shape (2, 3) with random floats between 0 and 1
rand_array = np.random.uniform(low=0.0, high=1.0, size=(2, 3))
print(rand_array)
```

Output:

```
[[0.52426917 0.64993811 0.53485944]
 [0.42440142 0.32023771 0.91741637]]
```

# Numpy statistics

Statistics is a field of study that uses data to make observations about populations (groups of objects). In statistics textbooks they are often called "distributions" instead of "populations". Probability is integral part of statistics.

Basic statistical operations include:

## Mean

$$\bar{\mu}=\frac{1}{N}\sum_{i=1}^N x_i$$

The mean is the sum of all values in the array divided by the total number of values.

```python
np.mean(arr)
```

## Median

Median is the value separating the higher half from the lower half of a data sample. When the number of observations is odd, it is the middle value. When the number of observations is even, it is the average of the two middle values.

```python
np.median(arr)
```

## Variance

Variance is a measure of how far a set of numbers is spread out from their average value. It is the average of the squared differences from the mean.

$$\sigma^2=\frac{1}{N}\sum_{i=1}^N(x_i-\bar{x})^2$$

```python
np.var(arr)
```

## Standard deviation

Standard deviation is a measure of the amount of variation or dispersion of a set of values. It is the square root of the variance.

$$\sigma=\sqrt{\frac{1}{N}\sum_{i=1}^N(x_i-\bar{x})^2}$$

```python
np.std(arr)
```

## Summary

| Operation | Function |
| --- | --- |
| mean |  `np.mean(arr)` |
| median | `np.median(arr)` | 
| variance |  `np.var(arr)` |
| standard deviation | `np.std(arr)` |

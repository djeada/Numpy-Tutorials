## Random Numbers with Numpy

NumPy's random module offers an assortment of functions to generate random numbers from various distributions. Whether you're simulating data or need randomness for an algorithm, this module has you covered.

### Generating Floats Between 0 and 1

The function `np.random.rand()` produces an array of random floating-point numbers uniformly distributed over `[0, 1)`.

Function Signature:

```Python
np.random.rand(d0, d1, ..., dn)
```

Parameters:
- $d0, d1,..., dn$: Dimensions of the returned array.

Example:

```Python
import numpy as np
rand_array = np.random.rand(2, 3)
print(rand_array)
```

Expected Output:

```
[[0.51749304 0.05537001 0.68478923]
 [0.62190377 0.40855834 0.89849802]]
```

### Generating Numbers from Standard Normal Distribution

`np.random.randn()` returns numbers from the standard normal distribution (mean = 0, standard deviation = 1).

Function Signature:

```Python
np.random.randn(d0, d1, ..., dn)
```

Example:

```Python
rand_norm_array = np.random.randn(2, 3)
print(rand_norm_array)
```

Expected Output:

```
[-1.20108323  0.45481233 -0.45698344]
 [ 0.34275595 -1.37612312  1.23458913]]
```

### Generating Random Integers

Use `np.random.randint()` to obtain random integers over a specified range.

Function Signature:

```Python
np.random.randint(low, high=None, size=None)
```

Parameters:
- low: The lowest integer in the range.
- high: One above the largest integer in the range.
- size: Output shape (default is a single value).

Example:

```Python
rand_integers = np.random.randint(0, 10, size=5)
print(rand_integers)
```

Expected Output:

```
[6 3 8 1 9]
```

### Generating Floats Over a Range

np.random.uniform() generates random floats over a specified range.

Function Signature:

```Python
np.random.uniform(low=0.0, high=1.0, size=None)
```

Example:

```Python
rand_uniform_array = np.random.uniform(0.5, 1.5, size=(2, 3))
print(rand_uniform_array)
```

Expected Output:

```
[[1.32149298 0.64893357 1.23158464]
 [1.10294322 0.95623745 1.48312411]]
```

## Statistics with Numpy

Statistics, at its core, is the science of collecting, analyzing, and interpreting data. It serves as a foundational pillar for fields such as data science, economics, and social sciences. A key component of statistics is understanding various distributions or, as some textbooks refer to them, populations. Central to this understanding is the idea of probability.

NumPy provides robust functions for a range of statistical operations, making it indispensable for data analysis in Python. Below, we explore some of these basic statistical operations.

### Mean

The mean or average of a set of values is computed by taking the sum of these values and dividing by the number of values.

$$ \bar{\mu} = \frac{1}{N} \sum_{i=1}^{N} x_i $$

Function:

```python
np.mean(arr)
```

### Median

The median is the middle value of an ordered set of values. For an odd number of values, it's the central value. For an even number of values, it's the average of the two middle values.

Function:

```python
np.median(arr)
```

### Variance

Variance quantifies the spread or dispersion of a set of values. It's calculated as the average of the squared differences of each value from the mean.

$$\sigma^2=\frac{1}{N}\sum_{i=1}^N(x_i-\bar{x})^2$$

Function:

```python
np.var(arr)
```

### Standard Deviation

The standard deviation measures the average distance between each data point and the mean. It's essentially the square root of variance.

$$\sigma=\sqrt{\frac{1}{N}\sum_{i=1}^N(x_i-\bar{x})^2}$$

Function:

```python
np.std(arr)
```

### Quick Reference Table for Statistical Operations in NumPy

| Operation           | Description                                       | Formula                                                       | NumPy Function     |
|---------------------|---------------------------------------------------|---------------------------------------------------------------|--------------------|
| Mean                | Average of values                                 | $\bar{\mu} = \frac{1}{N} \sum_{i=1}^{N} x_i$              | `np.mean(arr)`     |
| Median              | Middle value in an ordered set                    | -                                                             | `np.median(arr)`   |
| Variance            | Average of squared differences from the mean      | $\sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \bar{x})^2$  | `np.var(arr)`      |
| Standard Deviation  | Average distance of each point from the mean      | $\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_i - \bar{x})^2}$ | `np.std(arr)`      |


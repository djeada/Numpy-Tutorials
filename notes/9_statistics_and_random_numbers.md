## Random Numbers with NumPy

NumPy's random module is a powerful tool for generating random numbers from various distributions. Whether you are simulating data, implementing algorithms that require randomness, or performing statistical analysis, NumPy's random module has extensive capabilities to suit your needs.

### Generating Random Floats Between 0 and 1

The function `np.random.rand()` produces an array of random floating-point numbers uniformly distributed over the interval $[0, 1)$.

#### Function Signature:

```Python
np.random.rand(d0, d1, ..., dn)
```

#### Parameters:
- $d0, d1, ..., dn$: Dimensions of the returned array.

#### Example:

```Python
import numpy as np

rand_array = np.random.rand(2, 3)
print(rand_array)
```

#### Expected Output:

```
[[0.51749304 0.05537001 0.68478923]
 [0.62190377 0.40855834 0.89849802]]
```

### Generating Random Numbers from a Standard Normal Distribution

The function `np.random.randn()` returns numbers from the standard normal distribution, which has a mean of 0 and a standard deviation of 1.

#### Function Signature:

```Python
np.random.randn(d0, d1, ..., dn)
```

#### Example:

```Python
rand_norm_array = np.random.randn(2, 3)
print(rand_norm_array)
```

#### Expected Output:

```
[[-1.20108323  0.45481233 -0.45698344]
 [ 0.34275595 -1.37612312  1.23458913]]
```

### Generating Random Integers

The function `np.random.randint()` generates random integers from a specified range.

#### Function Signature:

```Python
np.random.randint(low, high=None, size=None)
```

#### Parameters:
- **low**: The lowest integer in the range.
- **high**: One above the largest integer in the range (exclusive).
- **size**: The shape of the output array (default is a single value).

#### Example:

```Python
rand_integers = np.random.randint(0, 10, size=5)
print(rand_integers)
```

#### Expected Output:

```
[6 3 8 1 9]
```

### Generating Random Floats Over a Specified Range

The function `np.random.uniform()` generates random floating-point numbers over a specified range $[low, high)$.

#### Function Signature:

```Python
np.random.uniform(low=0.0, high=1.0, size=None)
```

#### Example:

```Python
rand_uniform_array = np.random.uniform(0.5, 1.5, size=(2, 3))
print(rand_uniform_array)
```

#### Expected Output:

```
[[1.32149298 0.64893357 1.23158464]
 [1.10294322 0.95623745 1.48312411]]
```

### Generating Random Numbers from Other Distributions

NumPy also supports generating random numbers from other statistical distributions, such as binomial, Poisson, exponential, and many more.

#### Binomial Distribution

The function `np.random.binomial()` simulates the outcome of performing $n$ Bernoulli trials with success probability $p$.

```Python
np.random.binomial(n, p, size=None)
```

Example:

```Python
rand_binomial = np.random.binomial(10, 0.5, size=5)
print(rand_binomial)
```

Expected Output:

```
[4 5 6 7 5]
```

#### Poisson Distribution

The function `np.random.poisson()` generates random numbers from a Poisson distribution with a given mean $\lambda$.

```Python
np.random.poisson(lam, size=None)
```

Example:

```Python
rand_poisson = np.random.poisson(5, size=5)
print(rand_poisson)
```

Expected Output:

```
[3 4 7 2 6]
```

#### Exponential Distribution

The function `np.random.exponential()` generates random numbers from an exponential distribution with a specified scale parameter $\beta$.

```Python
np.random.exponential(scale=1.0, size=None)
```

Example:

```Python
rand_exponential = np.random.exponential(1.5, size=5)
print(rand_exponential)
```

Expected Output:

```
[0.35298273 1.8726912  0.73239216 2.51090448 1.2078675 ]
```

### Setting the Random Seed

To ensure reproducibility of random numbers, you can set the random seed using `np.random.seed()`. This is particularly useful for debugging or sharing code where you want others to generate the same sequence of random numbers.

#### Example:

```Python
np.random.seed(42)

# Generate random numbers
rand_array1 = np.random.rand(2, 3)
print(rand_array1)

# Reset seed and generate again
np.random.seed(42)
rand_array2 = np.random.rand(2, 3)
print(rand_array2)
```

#### Expected Output:

Both arrays will be identical because the seed was reset:

```
[[0.37454012 0.95071431 0.73199394]
 [0.59865848 0.15601864 0.15599452]]

[[0.37454012 0.95071431 0.73199394]
 [0.59865848 0.15601864 0.15599452]]
```

## Statistics with NumPy

Statistics, at its core, is the science of collecting, analyzing, and interpreting data. It serves as a foundational pillar for fields such as data science, economics, and social sciences. A key component of statistics is understanding various distributions or, as some textbooks refer to them, populations. Central to this understanding is the idea of probability.

NumPy provides robust functions for a range of statistical operations, making it indispensable for data analysis in Python. Below, we explore some of these basic and advanced statistical operations.

### Basic Statistical Measures

#### Mean

The mean or average of a set of values is computed by taking the sum of these values and dividing by the number of values.

$$ \bar{\mu} = \frac{1}{N} \sum_{i=1}^{N} x_i $$

Function:

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
mean_value = np.mean(arr)
print("Mean:", mean_value)
```

#### Median

The median is the middle value of an ordered set of values. For an odd number of values, it's the central value. For an even number of values, it's the average of the two middle values.

Function:

```python
median_value = np.median(arr)
print("Median:", median_value)
```

#### Variance

Variance quantifies the spread or dispersion of a set of values. It's calculated as the average of the squared differences of each value from the mean.

$$\sigma^2 = \frac{1}{N}\sum_{i=1}^N(x_i - \bar{x})^2$$

Function:

```python
variance_value = np.var(arr)
print("Variance:", variance_value)
```

#### Standard Deviation

The standard deviation measures the average distance between each data point and the mean. It's essentially the square root of variance.

$$\sigma = \sqrt{\frac{1}{N}\sum_{i=1}^N(x_i - \bar{x})^2}$$

Function:

```python
std_deviation = np.std(arr)
print("Standard Deviation:", std_deviation)
```

### Advanced Statistical Measures

#### Percentile

The percentile rank of a score is the percentage of scores in its frequency distribution that are equal to or lower than it. 

Function:

```python
percentile_50 = np.percentile(arr, 50)  # Median
print("50th Percentile (Median):", percentile_50)

percentile_90 = np.percentile(arr, 90)
print("90th Percentile:", percentile_90)
```

#### Quantile

Quantiles are values that divide a set of observations into equal parts. The 0.25 quantile is equivalent to the 25th percentile.

Function:

```python
quantile_value = np.quantile(arr, 0.25)
print("25th Quantile:", quantile_value)
```

#### Skewness

Skewness measures the asymmetry of the probability distribution of a real-valued random variable about its mean. 

Function:

```python
from scipy.stats import skew

skewness_value = skew(arr)
print("Skewness:", skewness_value)
```

#### Kurtosis

Kurtosis measures the "tailedness" of the probability distribution of a real-valued random variable. 

Function:

```python
from scipy.stats import kurtosis

kurtosis_value = kurtosis(arr)
print("Kurtosis:", kurtosis_value)
```

### Correlation

Correlation measures the relationship between two variables and ranges from -1 to 1.

Function:

```python
x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 4, 3, 2, 1])

correlation_matrix = np.corrcoef(x, y)
print("Correlation matrix:\n", correlation_matrix)
```

### Covariance

Covariance indicates the direction of the linear relationship between variables.

Function:

```python
covariance_matrix = np.cov(x, y)
print("Covariance matrix:\n", covariance_matrix)
```

### Applying Statistics to Multidimensional Data

NumPy allows statistical operations on multidimensional data along specified axes. 

#### Example:

```python
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

mean_rows = np.mean(data, axis=1)
print("Mean of each row:", mean_rows)

mean_columns = np.mean(data, axis=0)
print("Mean of each column:", mean_columns)
```

### Example Application: Descriptive Statistics

To showcase how these statistical functions can be applied in practice, let’s calculate various descriptive statistics for a given dataset.

```python
# Sample dataset
data = np.random.normal(0, 1, 1000)

# Mean
mean = np.mean(data)
print("Mean:", mean)

# Median
median = np.median(data)
print("Median:", median)

# Variance
variance = np.var(data)
print("Variance:", variance)

# Standard Deviation
std_dev = np.std(data)
print("Standard Deviation:", std_dev)

# 25th and 75th Percentiles
q25 = np.percentile(data, 25)
q75 = np.percentile(data, 75)
print("25th Percentile:", q25)
print("75th Percentile:", q75)

# Skewness
skewness = skew(data)
print("Skewness:", skewness)

# Kurtosis
kurt = kurtosis(data)
print("Kurtosis:", kurt)
```

### Quick Reference Table for Statistical Operations in NumPy

| Operation           | Description                                       | Formula                                                       | NumPy Function     |
|---------------------|---------------------------------------------------|---------------------------------------------------------------|--------------------|
| Mean                | Average of values                                 | $\bar{\mu} = \frac{1}{N} \sum_{i=1}^{N} x_i$              | `np.mean(arr)`     |
| Median              | Middle value in an ordered set                    | -                                                             | `np.median(arr)`   |
| Variance            | Average of squared differences from the mean      | $\sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \bar{x})^2$  | `np.var(arr)`      |
| Standard Deviation  | Average distance of each point from the mean      | $\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_i - \bar{x})^2}$ | `np.std(arr)`      |


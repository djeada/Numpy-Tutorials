## Random Numbers and Statistics

Statistics, at its core, is the science of collecting, analyzing, and interpreting data. It serves as a foundational pillar for fields such as data science, economics, and social sciences. An important component of statistics is understanding various distributions or, as some textbooks refer to them, populations. Central to this understanding is the idea of probability.

### Generating Random Floats Between 0 and 1

The function `np.random.rand()` produces an array of random floating-point numbers uniformly distributed over the interval $[0, 1)$.

Function Signature:

```Python
np.random.rand(d0, d1, ..., dn)
```

Parameters:

- $d0, d1, ..., dn$: Dimensions of the returned array.

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

### Generating Random Numbers from a Standard Normal Distribution

The function `np.random.randn()` returns numbers from the standard normal distribution, which has a mean of 0 and a standard deviation of 1.

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
[[-1.20108323  0.45481233 -0.45698344]
 [ 0.34275595 -1.37612312  1.23458913]]
```

### Generating Random Integers

The function `np.random.randint()` generates random integers from a specified range.

Function Signature:

```Python
np.random.randint(low, high=None, size=None)
```

Parameters:

- **low** is the parameter that represents the smallest integer in the range.
- **high** is the parameter that defines the upper bound of the range, but it is exclusive, meaning the value is one above the largest possible integer.
- **size** is the parameter that determines the shape of the output array, with the default being a single value.

Example:

```Python
rand_integers = np.random.randint(0, 10, size=5)
print(rand_integers)
```

Expected Output:

```
[6 3 8 1 9]
```

### Generating Random Floats Over a Specified Range

The function `np.random.uniform()` generates random floating-point numbers over a specified range $[low, high)$.

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

Example:

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

Expected Output:

Both arrays will be identical because the seed was reset:

```
[[0.37454012 0.95071431 0.73199394]
 [0.59865848 0.15601864 0.15599452]]

[[0.37454012 0.95071431 0.73199394]
 [0.59865848 0.15601864 0.15599452]]
```

### Basic Statistical Measures

The basic statistical measures provide insights into the central tendency and variability of a dataset. These metrics form the foundation for more advanced analyses and are crucial for summarizing data.

#### Mean

The mean or arithmetic average of a dataset represents its central point. It is defined as the sum of all observations divided by the number of observations. This measure is widely used due to its simplicity, but it can be influenced by extreme values.

$$
\mu = \frac{1}{N} \sum_{i=1}^{N} x_i
$$

Function:

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
mean_value = np.mean(arr)
print("Mean:", mean_value)
```

Expected output:

```
Mean: 3.0
```

* `arr`: Defines a NumPy array with values 1 through 5.
* `np.mean(arr)`: Computes the arithmetic mean of the array.
* `print(...)`: Displays the result.

#### Median

The median is the middle value of an ordered dataset, providing a robust measure of central tendency that is less sensitive to outliers.

Function:

```python
median_value = np.median(arr)
print("Median:", median_value)
```

Expected output:

```
Median: 3.0
```

* `np.median(arr)`: Calculates the median value.
* The median splits the dataset into two equal halves.

#### Variance

Variance quantifies the spread of data points around the mean. It is the average of the squared deviations from the mean.

$$
\sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2
$$

Function:

```python
variance_value = np.var(arr)
print("Variance:", variance_value)
```

Expected output:

```
Variance: 2.0
```

* `np.var(arr)`: Computes the population variance.
* Squared deviations amplify the effect of larger differences.

#### Standard Deviation

Standard deviation is the square root of the variance and expresses dispersion in the same units as the data.

$$
\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2}
$$

Function:

```python
std_deviation = np.std(arr)
print("Standard Deviation:", std_deviation)
```

Expected output:

```
Standard Deviation: 1.4142135623730951
```

* `np.std(arr)`: Calculates the population standard deviation.
* The result indicates average distance from the mean.

### Advanced Statistical Measures

Advanced measures provide deeper insights into distribution shape and position relative to other data.

#### Percentile

A percentile indicates the value below which a given percentage of observations fall. The 50th percentile is the median.

Function:

```python
percentile_50 = np.percentile(arr, 50)  # Median
print("50th Percentile (Median):", percentile_50)

percentile_90 = np.percentile(arr, 90)
print("90th Percentile:", percentile_90)
```

Expected output:

```
50th Percentile (Median): 3.0
90th Percentile: 5.0
```

* `np.percentile(arr, p)`: Computes the p-th percentile.
* 90th percentile shows the value below which 90% of data lie.

#### Quantile

Quantiles divide data into equal-sized, contiguous intervals. The p-th quantile corresponds to the (p\*100)th percentile.

Function:

```python
quantile_value = np.quantile(arr, 0.25)
print("25th Quantile:", quantile_value)
```

Expected output:

```
25th Quantile: 2.0
```

* `np.quantile(arr, q)`: Finds the q-th quantile.
* 0.25 quantile marks the first quartile.

#### Skewness

Skewness measures asymmetry of the distribution. Positive skew indicates a longer right tail.

Function:

```python
from scipy.stats import skew

skewness_value = skew(arr)
print("Skewness:", skewness_value)
```

Expected output:

```
Skewness: 0.0
```

* `skew(arr)`: Calculates sample skewness.
* A zero skewness denotes a symmetric distribution.

#### Kurtosis

Kurtosis reflects the 'tailedness' of a distribution. Higher values indicate heavier tails.

Function:

```python
from scipy.stats import kurtosis

kurtosis_value = kurtosis(arr)
print("Kurtosis:", kurtosis_value)
```

Expected output:

```
Kurtosis: -1.2
```

* `kurtosis(arr)`: Computes excess kurtosis.
* Negative kurtosis shows lighter tails than a normal distribution.

### Correlation

Correlation quantifies linear relationship strength between two variables, ranging from -1 to 1.

Function:

```python
x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 4, 3, 2, 1])

correlation_matrix = np.corrcoef(x, y)
print("Correlation matrix:", correlation_matrix)
```

Expected output:

```
Correlation matrix:
[[ 1.  -1. ]
 [ -1.   1. ]]
```

* `np.corrcoef(x, y)`: Returns the correlation matrix.
* Values of -1 indicate a perfect negative correlation.

### Covariance

Covariance indicates the direction of linear relationship; positive values show same direction movement.

Function:

```python
covariance_matrix = np.cov(x, y)
print("Covariance matrix:", covariance_matrix)
```

Expected output:

```
Covariance matrix:
[[ 2.5 -2.5]
 [-2.5  2.5]]
```

* `np.cov(x, y)`: Computes covariance matrix.
* Off-diagonal elements represent covariance between x and y.

### Applying Statistics to Multidimensional Data

NumPy extends statistical operations along specified axes for matrices and higher-dimensional arrays.

Example:

```python
data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

mean_rows = np.mean(data, axis=1)
print("Mean of each row:", mean_rows)

mean_columns = np.mean(data, axis=0)
print("Mean of each column:", mean_columns)
```

Expected output:

```
Mean of each row: [2. 5. 8.]
Mean of each column: [4. 5. 6.]
```

* `axis=1`: Computes mean across columns for each row.
* `axis=0`: Computes mean across rows for each column.

### Example Application: Descriptive Statistics

Below is a practical application computing descriptive statistics for a normally distributed sample.

```python
import numpy as np
from scipy.stats import skew, kurtosis

# Sample dataset: 1000 points from a N(0,1)
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

Expected output (values will vary):

```
Mean: ~0.0
Median: ~0.0
Variance: ~1.0
Standard Deviation: ~1.0
25th Percentile: ~-0.674
75th Percentile: ~0.674
Skewness: ~0.0
Kurtosis: ~0.0
```

* `np.random.normal(0, 1, 1000)`: Generates random sample from standard normal distribution.
* Subsequent functions compute descriptive metrics.
* Tilde (\~) indicates approximate values due to randomness.
  
### Reference Table

| Operation                  | Description                                                                                                                                                                     | Formula†                                                  | NumPy Call                    | Example                                                                           | Expected Output    |
|----------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------|-------------------------------|-----------------------------------------------------------------------------------|--------------------|
| **Mean**                   | Arithmetic average                                                                                                                                                              | $\displaystyle \bar x = \frac{1}{N}\sum_{i=1}^N x_i$       | `np.mean(a)`                  | <code>import numpy as np<br>a = np.array([1,2,3,4,5])<br>np.mean(a)</code>        | `3.0`              |
| **Median**                 | Middle value (50th percentile)                                                                                                                                                  | $\displaystyle x_{(0.5)}$                                 | `np.median(a)`                | `np.median(a)`                                                                   | `3.0`              |
| **Variance**               | Avg. squared deviation<br>• *Pop.* (`ddof=0`): $\displaystyle \sigma^2=\frac{1}{N}\sum(x_i-\bar x)^2$ <br>• *Sample* (`ddof=1`): $\displaystyle s^2=\frac{1}{n-1}\sum(x_i-\bar x)^2$ | `np.var(a,ddof=0)`<br>`np.var(a,ddof=1)`                  | `np.var(a)`                   | `2.0`                                                                            |                    |
| **Std. deviation**         | Square-root of variance                                                                                                                                                         | $\displaystyle \sigma=\sqrt{\sigma^2}$ (or $s$)           | `np.std(a,ddof=0)`             | `np.std(a)`                                                                      | `1.414213562…`     |
| **Minimum / Maximum**      | Smallest / largest element                                                                                                                                                      | $\displaystyle \min(x)$ / $\max(x)$                       | `np.min(a)` / `np.max(a)`     | `np.min(a)`                                                                      | `1`                |
| **Range (ptp)**            | Peak-to-peak span                                                                                                                                                               | $\displaystyle \max(x) - \min(x)$                         | `np.ptp(a)`                   | `np.ptp(a)`                                                                      | `4`                |
| **Sum / Product**          | Add / multiply all values                                                                                                                                                       | $\displaystyle \sum_i x_i$ / $\displaystyle \prod_i x_i$ | `np.sum(a)` / `np.prod(a)`    | `np.sum(a)`                                                                      | `15`               |
| **Cumulative Sum / Prod.** | Running total / product                                                                                                                                                         | $S_k=\sum_{i\le k}x_i$<br>$P_k=\prod_{i\le k}x_i$          | `np.cumsum(a)` / `np.cumprod(a)` | `np.cumsum(a)`                                                                  | `[ 1  3  6 10 15]` |
| **Percentile**             | Value below which *q* % of data lie                                                                                                                                             | $\displaystyle x_{(q/100)}$                                | `np.percentile(a,q)`          | `np.percentile(a,50)`                                                            | `3.0`              |
| **Correlation** $\rho$     | Linear relationship (–1 … 1)                                                                                                                                                    | $\displaystyle \rho_{xy}=\frac{\mathrm{cov}(x,y)}{\sigma_x\,\sigma_y}$ | `np.corrcoef(x,y)`            | <code>import numpy as np<br>x = np.array([1,2,3])<br>y = np.array([4,5,6])<br>np.corrcoef(x,y)</code> | `[[1. 1.]\n [1. 1.]]` |
| **Covariance**             | Co-variation of two variables                                                                                                                                                    | $\displaystyle \mathrm{cov}(x,y)=\frac{1}{n-1}\sum(x_i-\bar x)(y_i-\bar y)$ | `np.cov(x,y)`                 | `np.cov(x,y)`                                                                   | `[[1. 1.]\n [1. 1.]]` |


<sup>† Formulas assume a one-dimensional population of size `N` (or sample of size `n`). NumPy’s `ddof` handles population vs sample.</sup>

*Matrix note:* `np.corrcoef` and `np.cov` return a 2×2 matrix when given two 1-D arrays: row/column 0 is `x`, row/column 1 is `y` (diagonals = self-correlation/variance).

## Random Numbers and Statistics

Statistics, at its core, is the science of collecting, analyzing, and interpreting data. It serves as a foundational pillar for fields such as data science, economics, and social sciences. An important component of statistics is understanding various distributions or, as some textbooks refer to them, populations. Central to this understanding is the idea of probability.

### Random Number Generation

NumPy’s random module offers a comprehensive suite of functions for generating pseudorandom numbers across a variety of distributions and data types. Whether you need uniformly distributed floats in $[0, 1)$, samples from a standard normal distribution, random integers within a specified range, or values drawn from binomial, Poisson, exponential, and many other statistical distributions, NumPy provides intuitive, high-performance methods to meet your needs. Additionally, setting a seed via `np.random.seed()` ensures that your experiments and simulations are reproducible, making it easy to debug and share your work. With simple, consistent function signatures and flexible array-shaping capabilities, NumPy allows you to incorporate randomness into machine learning initializations, Monte Carlo simulations, synthetic data generation, and beyond.

#### Generating Random Floats Between 0 and 1

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

#### Generating Random Numbers from a Standard Normal Distribution

The **normal distribution** (or Gaussian distribution) is a continuous probability distribution for a real-valued random variable $X$, whose probability density function is

$$
f(x; \mu, \sigma) \;=\; \frac{1}{\sigma\sqrt{2\pi}}
\exp\!\Bigl(-\frac{(x - \mu)^2}{2\sigma^2}\Bigr)
$$

where $\mu$ is the mean and $\sigma$ the standard deviation.  A **standard normal distribution** is the special case with $\mu = 0$ and $\sigma = 1$, often denoted $N(0,1)$.

In this context, **generation** means **sampling**—i.e.\ drawing independent random values whose statistical behavior matches that of the target distribution.  NumPy’s `np.random.randn()` does exactly that for the standard normal.

Function Signature:

```python
np.random.randn(d0, d1, ..., dn)
```

* **d0, d1, …, dn**: dimensions of the returned array of samples.

Example:

```python
import numpy as np

# Draw 2×3 independent samples from a standard normal
rand_norm_array = np.random.randn(2, 3)
print(rand_norm_array)
```

Expected Output:

```
[[-1.20108323  0.45481233 -0.45698344]
 [ 0.34275595 -1.37612312  1.23458913]]
```


#### Generating Random Integers

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

#### Generating Random Floats Over a Specified Range

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

#### Binomial Distribution

The **binomial distribution** describes the probability of obtaining exactly $k$ successes in $n$ independent Bernoulli trials (each trial with success probability $p$).  Its probability mass function is

$$
P(X = k) \;=\; \binom{n}{k} p^k (1 - p)^{\,n - k}
$$

for $k = 0, 1, \dots, n$.  **Generation** here means **sampling** a collection of independent draws $X_1, X_2, \dots$ from this distribution.

```python
np.random.binomial(n, p, size=None)
```

* **n**: number of trials
* **p**: success probability per trial
* **size**: shape of the output array of independent samples

Example:

```python
import numpy as np

# Simulate 5 independent draws of the number of successes in 10 trials with p=0.5
rand_binomial = np.random.binomial(10, 0.5, size=5)
print(rand_binomial)
```

Expected Output (your values will vary):

```
[4 5 6 7 5]
```

#### Poisson Distribution

The **Poisson distribution** models the count of events occurring in a fixed interval of time or space when these events happen with a known constant rate $\lambda$ and independently of the time since the last event.  Its probability mass function is

$$
P(X = k) \;=\; \frac{\lambda^k e^{-\lambda}}{k!}
$$

for $k = 0, 1, 2, \dots$.  **Generation** means drawing samples whose frequencies of occurrence match this distribution.

```python
np.random.poisson(lam, size=None)
```

* **lam**: the expected number of events (rate $\lambda$)
* **size**: shape of the output array of independent samples

Example:

```python
rand_poisson = np.random.poisson(5, size=5)
print(rand_poisson)
```

Expected Output:

```
[3 4 7 2 6]
```

#### Exponential Distribution

The **exponential distribution** is a continuous distribution often used to model waiting times between independent events that occur at a constant average rate.  Its probability density function is

$$
f(x; \beta) \;=\; \frac{1}{\beta} \exp\!\biggl(-\frac{x}{\beta}\biggr)
$$

for $x \ge 0$ and scale parameter $\beta > 0$.  **Generation** refers to sampling real values whose distribution of inter-arrival times follows this law.

```python
np.random.exponential(scale=1.0, size=None)
```

* **scale** ($\beta$): the mean waiting time
* **size**: shape of the output array of independent samples

Example:

```python
rand_exponential = np.random.exponential(1.5, size=5)
print(rand_exponential)
```

Expected Output:

```
[0.35298273 1.87269120 0.73239216 2.51090448 1.20786750]
```

#### Setting the Random Seed

A pseudorandom number generator (PRNG) in NumPy uses an internal **state vector** and a deterministic algorithm (by default, the Mersenne Twister) to produce a sequence of values.  **Seeding** means initializing that state vector from a single integer $s$, so that all subsequent “random” outputs are **fully determined** by $s$.  Mathematically, if you denote the PRNG’s state-update and output function as

$$
\text{state}_{i+1},\,x_i \;=\; F(\text{state}_i)
$$

then setting $\text{state}_0 = \text{Init}(s)$ ensures the sequence $\{x_0, x_1, \dots\}$ is reproducible.

```python
np.random.seed(seed)
```

* **seed**: any nonnegative integer used to initialize the PRNG’s state.

Example: Deterministic Uniform Floats

```python
import numpy as np

# Initialize the PRNG with seed=123
np.random.seed(123)

# Generate two uniform [0,1) samples
a = np.random.rand(2)
print(a)

# Re-seed and generate again
np.random.seed(123)
b = np.random.rand(2)
print(b)
```

Expected Output:

```
[0.69646919 0.28613933]
[0.69646919 0.28613933]
```

Here, both `a` and `b` are identical because the same seed leads to the same initial state and thus the same outputs.

You can apply the same principle to any other NumPy random function—be it `randn()`, `randint()`, `binomial()`, etc.—to obtain reproducible sequences in simulations, experiments, or unit tests.

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

| Operation                  | Formula†                                                                                                                                                        | NumPy Call                               | Example                                                                                                 | Expected Output       |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------- | --------------------- |
| **Mean**                   | $\bar x = \frac{1}{N}\sum\_{i=1}^N x\_i$                                                                                                        | `np.mean(a)`                             | <code>import numpy as np<br>a = np.array(\[1,2,3,4,5])<br>np.mean(a)</code>                             | `3.0`                 |
| **Median**                 | $x\_{(0.5)}$                                                                                                                                    | `np.median(a)`                           | `np.median(a)`                                                                                          | `3.0`                 |
| **Variance**               | • *Pop.* (`ddof=0`): $ \sigma^2=\frac{1}{N}\sum(x\_i-\bar x)^2$<br>• *Sample* (`ddof=1`): $ s^2=\frac{1}{n-1}\sum(x\_i-\bar x)^2$ | `np.var(a,ddof=0)`<br>`np.var(a,ddof=1)` | `np.var(a)`                                                                                             |                       |
| **Std. deviation**         | $\sigma=\sqrt{\sigma^2}$ (or $s$)                                                                                                             | `np.std(a,ddof=0)`                       | `np.std(a)`                                                                                             | `1.414213562…`        |
| **Minimum / Maximum**      | $\min(x)$ / $\max(x)$                                                                                                                         | `np.min(a)` / `np.max(a)`                | `np.min(a)`                                                                                             | `1`                   |
| **Range (ptp)**            | $\max(x) - \min(x)$                                                                                                                             | `np.ptp(a)`                              | `np.ptp(a)`                                                                                             | `4`                   |
| **Sum / Product**          | $\sum\_i x\_i$ / $ \prod\_i x\_i$                                                                                                | `np.sum(a)` / `np.prod(a)`               | `np.sum(a)`                                                                                             | `15`                  |
| **Cumulative Sum / Prod.** | $S\_k=\sum\_{i\le k}x\_i$<br>$P\_k=\prod\_{i\le k}x\_i$                                                                                                     | `np.cumsum(a)` / `np.cumprod(a)`         | `np.cumsum(a)`                                                                                          | `[ 1  3  6 10 15]`    |
| **Percentile**             | $x\_{(q/100)}$                                                                                                                                  | `np.percentile(a,q)`                     | `np.percentile(a,50)`                                                                                   | `3.0`                 |
| **Correlation** $\rho$   | $\rho\_{xy}=\frac{\mathrm{cov}(x,y)}{\sigma\_x,\sigma\_y}$                                                                                      | `np.corrcoef(x,y)`                       | <code>import numpy as np<br>x = np.array(\[1,2,3])<br>y = np.array(\[4,5,6])<br>np.corrcoef(x,y)</code> | `[[1. 1.]\n [1. 1.]]` |
| **Covariance**             | $\mathrm{cov}(x,y)=\frac{1}{n-1}\sum(x\_i-\bar x)(y\_i-\bar y)$                                                                                 | `np.cov(x,y)`                            | `np.cov(x,y)`                                                                                           | `[[1. 1.]\n [1. 1.]]` |

<sup>† Formulas assume a one-dimensional population of size `N` (or sample of size `n`). NumPy’s `ddof` handles population vs sample.</sup>

*Matrix note:* `np.corrcoef` and `np.cov` return a 2×2 matrix when given two 1-D arrays: row/column 0 is `x`, row/column 1 is `y` (diagonals = self-correlation/variance).

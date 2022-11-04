
<h1>Random numbers</h1>

1. Floats between 0 and 1.

```Python
np.random.rand(d0, d1...)
```
  It generate an array with random numbers (float) that are uniformly distributed between 0 and 1.
  The parameter allows you to specify the shape of the array.
  
2. Standard normal distribution.

```Python
np.random.randn(d0, d1...)
```
  It generates an array with random numbers (float) that are normally distrbuted. Mean = 0, Stdev (standard deviation) = 1.
  
3. Random integers within range

```Python
np.random.randint(low, high=None, size=None)
```

It generates an array with random numbers (integers) that are uniformly distributed between 0 and given number.

4. Random floats within range

```Python
np.random.uniform(low=0.0, high=1.0, size=None)
```

It generates an array with random numbers (float) between given numbers.

<h1>Numpy statistics</h1>

Statistics is a field of study that uses data to make observations about populations (groups of objects). In statistics textbooks they are often called "distributions" instead of "populations". Probability is integral part of statistics.

Basic statistical operations include:

1. Mean

$$\bar{\mu}=\frac{1}{N}\sum_{i=1}^N x_i$$

2. Median

Median formula when $N$ is odd:

$$m = x_{\frac{N + 1}{2}} $$

Median formula when $N$ is even:

$$m = \frac{x_{\frac{N}{2}} + x_{\frac{N}{2}+1}}{2}$$
  
3. Variance

$$\sigma^2=\frac{1}{N}\sum_{i=1}^N(x_i-\bar{x})^2$$

4. Standard deviation

$$\sigma=\sqrt{\frac{1}{N}\sum_{i=1}^N(x_i-\bar{x})^2}$$

| Operation | Function |
| --- | --- |
| mean |  np.mean(arr) |
| median | np.median(arr) | 
| variance |  np.var(arr) |
| standard deviation | np.std(arr) | |

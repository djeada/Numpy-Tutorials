# Statistics and random numbers: state the quantity being estimated

A statistic summarizes the data you actually have. A probability model describes
how data might be generated. A sample's mean is not required to equal its
model's mean, and changing an axis, denominator, or weighting rule can change
which quantity you are computing.

## Mean, median, variance, and standard deviation with one dataset

For $x=(2,4,4,4,5,5,7,9)$, $n=8$ and $\bar x=5$.
The sum of squared deviations is

$$
(2-5)^2+3(4-5)^2+2(5-5)^2+(7-5)^2+(9-5)^2=32.
$$

NumPy's variance divisor is $n-ddof$:

$$
\operatorname{var}(x,ddof)=\frac{\sum_i(x_i-\bar x)^2}{n-ddof}.
$$

```python
import numpy as np

x = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
print(float(x.mean()), float(np.median(x)))
print(float(np.sum((x - x.mean()) ** 2)))
print(float(x.var()), round(float(x.var(ddof=1)), 6))
print(float(x.std()), round(float(x.std(ddof=1)), 6))
```

```text
5.0 4.5
32.0
4.0 4.571429
2.0 2.13809
```

The median is $(4+5)/2=4.5$. The population variance of these eight values is
$32/8=4$; the sample variance with Bessel's correction is $32/7$.
Standard deviation is a root-mean-square deviation, not the mean absolute
distance from the mean.

For independent, identically distributed observations with finite variance,
`ddof=1` makes the sample **variance** an unbiased estimator of the population
variance. Taking its square root does not make the sample standard deviation
unbiased. Also require $n>ddof$; a one-observation sample does not have a
defined `ddof=1` variance.

## Accumulator precision and input precision are separate issues

Floating inputs generally retain their dtype in reductions such as `mean`.
A wider accumulator can prevent loss during addition, but cannot restore
information already rounded away when the input was stored.

```python
import numpy as np

x = np.array([1e8, 1.0, -1e8], dtype=np.float32)
print(float(np.sum(x)))
print(float(np.sum(x, dtype=np.float64)))

already_rounded = np.array([1e8, 1e8 + 1], dtype=np.float32)
print(already_rounded.astype(np.float64).tolist())
```

```text
0.0
1.0
[100000000.0, 100000000.0]
```

In the first sum, the intermediate $10^8+1$ is rounded back to $10^8$ in
`float32`. The exact input values can still be summed in `float64`.
In the second array, the two original inputs have already become identical;
changing dtype afterward cannot separate them.

The order and grouping of floating-point sums can affect results. A wider dtype
improves this example; it does not make every reduction exact.

## Reduce the observation axis to get one statistic per feature

Let rows be observations and columns be features. Then

$$
\mu_j=\frac1n\sum_iX_{ij},\qquad
Z_{ij}=\frac{X_{ij}-\mu_j}{s_j}.
$$

Reduce `axis=0` to summarize observations separately for each feature.
Reduce `axis=1` to summarize features separately for each observation.
Those answer different questions.

```python
import numpy as np

X = np.array([[1.0, 10.0], [3.0, 10.0], [5.0, 10.0]])
mean = X.mean(axis=0, keepdims=True)
spread = X.std(axis=0, keepdims=True)
centered = X - mean
Z = np.divide(centered, spread, out=np.zeros_like(centered), where=spread > 0)

print(mean.shape, spread.shape)
print(mean.ravel().tolist())
print(np.round(spread.ravel(), 6).tolist())
print(np.round(Z, 6).tolist())
print(np.allclose(Z.mean(axis=0), 0, atol=1e-12))
print(np.round(Z.var(axis=0), 6).tolist())
```

```text
(1, 2) (1, 2)
[3.0, 10.0]
[1.632993, 0.0]
[[-1.224745, 0.0], [0.0, 0.0], [1.224745, 0.0]]
True
[1.0, 0.0]
```

The second feature is constant, so unit-variance scaling is undefined.
This code explicitly maps it to zero; its resulting variance remains zero.
Adding a small epsilon to every denominator is another policy, but changes
nonconstant features too.

`keepdims=True` leaves shape `(1,2)` for broadcasting.
When preparing train/test data, estimate means and scales on the training data,
then apply those same values to test data; recomputing them on the test set
changes the transformation using test information.

## Missing values and unequal weights change the sample being summarized

`nanmean` ignores NaNs independently in each reduced slice. The result can use
different observations for different features.
Dropping incomplete rows is a different rule, with a different answer.

```python
import numpy as np

X = np.array([[1.0, np.nan], [3.0, 5.0]])
print(np.nanmean(X, axis=0).tolist())
complete = X[np.isfinite(X).all(axis=1)]
print(complete.mean(axis=0).tolist())

group_mean = np.array([90.0, 50.0])
group_size = np.array([1, 9])
print(float(group_mean.mean()))
print(float(np.average(group_mean, weights=group_size)))
```

```text
[2.0, 5.0]
[3.0, 5.0]
70.0
54.0
```

For the first feature, ignoring missing entries separately uses values $1$ and $3$;
complete-row filtering uses only $3$.
Ignoring missingness does not establish that the remaining data are representative.
An all-NaN slice has no mean and produces a warning and NaN; choose an explicit
policy for that case. `nanmean` also does not ignore infinities.

For the groups, the combined mean is

$$
\frac{1\cdot90+9\cdot50}{1+9}=54,
$$

whereas averaging the two group means gives each group equal weight and yields
$70$. Averaging averages is valid for the combined individual mean only when
the weights reflect the numbers of observations, or all groups are equally sized.

## Quantiles can be values that never occurred

NumPy's default `method="linear"` interpolates sorted data.
For $q\in[0,1]$, let $h=(n-1)q$, $j=\lfloor h\rfloor$, and $\alpha=h-j$.
Between neighbors, the quantile is

$$
Q(q)=(1-\alpha)x_{(j)}+\alpha x_{(j+1)}.
$$

At $q=1$, return the largest observation directly.
For `[0,10,20,100]` at $q=0.25$, $h=0.75$, so the answer is
$0.25\cdot0+0.75\cdot10=7.5$.

```python
import numpy as np

x = np.array([0.0, 10.0, 20.0, 100.0])
print(float(np.quantile(x, 0.25, method="linear")))
print(float(np.quantile(x, 0.25, method="lower")))
print(float(np.percentile(x, 25, method="linear")))
print(float(np.median(x)))
```

```text
7.5
0.0
7.5
15.0
```

`quantile` uses a fraction in $[0,1]$; `percentile` uses a percentage in
$[0,100]$. Different interpolation or estimation methods can legitimately return
different finite-sample quantiles. State the method when matching another tool
or a published calculation.

## Covariance depends on units; correlation measures linear association

For observations in rows, the sample covariance matrix is

$$
C=\frac{(X-\mathbf1\mu^\top)^\top(X-\mathbf1\mu^\top)}{n-1}.
$$

Its diagonal contains sample variances.
For nonconstant variables, correlation normalizes covariance:

$$
\rho_{jk}=\frac{C_{jk}}{\sqrt{C_{jj}}\sqrt{C_{kk}}}.
$$

```python
import numpy as np

X = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
centered = X - X.mean(axis=0)
C = np.cov(X, rowvar=False)  # columns are variables; default divisor n-1
print(C.tolist())
print(np.round(X.var(axis=0), 6).tolist())
print(X.var(axis=0, ddof=1).tolist())
print(np.round(np.corrcoef(X, rowvar=False), 6).tolist())
np.testing.assert_allclose(C, centered.T @ centered / (len(X) - 1))

scaled = X.copy()
scaled[:, 1] *= 100
print(np.cov(scaled, rowvar=False).tolist())
print(np.allclose(np.corrcoef(scaled, rowvar=False),
                  np.corrcoef(X, rowvar=False)))

x = np.array([-1.0, 0.0, 1.0])
y = x**2
print(round(float(np.corrcoef(x, y)[0, 1]), 6))
```

```text
[[1.0, 2.0], [2.0, 4.0]]
[0.666667, 2.666667]
[1.0, 4.0]
[[1.0, 1.0], [1.0, 1.0]]
[[1.0, 200.0], [200.0, 40000.0]]
True
0.0
```

`np.cov` defaults to treating **rows** as variables, unlike the observation-row
convention used here; set `rowvar=False` explicitly.
Its default divisor is $n-1$, while `np.var` defaults to $n$.
Comparing the covariance diagonal to `var` therefore requires `ddof=1`.

Multiplying the second variable by $100$ multiplies its variance by $100^2$
and its covariance with the first by $100$. Correlation stays the same under
positive rescaling.

In the last dataset, $y$ is completely determined by $x$, yet correlation is
zero: the relationship is nonlinear and symmetric. Zero correlation does not
imply independence, and correlation alone does not establish causation.
For a constant variable the denominator is zero, so correlation is undefined.

## Skewness and kurtosis need definitions, not just function names

Let $m_k=\frac1n\sum_i(x_i-\bar x)^k$. The uncorrected moment coefficients are

$$
g_1=\frac{m_3}{m_2^{3/2}},\qquad
g_2=\frac{m_4}{m_2^2}-3.
$$

$g_2$ is **excess** kurtosis; the subtraction makes the normal distribution's
population value zero. Some tools instead report $m_4/m_2^2$, whose normal
reference is $3$, or apply finite-sample corrections.

```python
import numpy as np

x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
d = x - x.mean()
m2 = np.mean(d**2)
m3 = np.mean(d**3)
m4 = np.mean(d**4)
skewness = m3 / m2**1.5
excess_kurtosis = m4 / m2**2 - 3
print(float(m2), round(float(m4), 6))
print(round(float(skewness), 6), round(float(excess_kurtosis), 6))
```

```text
2.0 6.8
0.0 -1.3
```

These are descriptive moment coefficients, not bias-corrected estimators.
They are undefined for constant data where $m_2=0$.
Symmetry implies zero third central moment when that moment exists, but zero
skewness alone does not prove symmetry. Kurtosis is sensitive to extreme
deviations because of the fourth power; interpreting it merely as “peak height”
is misleading. These definitions require only NumPy, not an undeclared SciPy import.

## A local random generator owns a stream that advances

A pseudorandom generator has state. A seed initializes that state; each draw
advances it. Recreating a generator with the same seed restarts the stream.
Making multiple draws from the same generator does not restart it.

```python
import numpy as np

rng = np.random.default_rng(123)
first = rng.random(3)
second = rng.random(3)
replay = np.random.default_rng(123).random(3)
print(np.array_equal(first, replay))
print(np.array_equal(first, second))

dice = rng.integers(1, 7, size=(2, 3))  # low inclusive, high exclusive
print(dice.shape)
print(bool(np.all((1 <= dice) & (dice <= 6))))

children = np.random.SeedSequence(123).spawn(2)
stream_a, stream_b = [np.random.default_rng(child) for child in children]
print(np.array_equal(stream_a.random(4), stream_b.random(4)))
```

```text
True
False
(2, 3)
True
False
```

Pass a generator into functions that need randomness. Creating
`default_rng(123)` inside every function call restarts the same sequence each time.
Spawning child streams is useful for reproducibly separating simulation runs;
initializing each worker with the identical seed instead gives duplicated streams.

`np.random.seed` controls the legacy global random state; it does not reset
an existing `Generator`. The global `rand`, `randn`, and `randint` APIs are
legacy alternatives to `rng.random`, `rng.standard_normal`, and `rng.integers`.

A seed alone is not a promise of identical distribution samples across every
NumPy version. Record the version, generator, draw order, and parameters when
exact replay matters. A finite sample also need not have exactly the model's
mean, variance, or proportions.

## Distribution parameters have units and distinct meanings

| Model | Generator call | Mean | Variance |
| --- | --- | --- | --- |
| Uniform on $[0,1)$ | `rng.random(size=n)` | $1/2$ | $1/12$ |
| Normal | `rng.normal(loc=mu, scale=sigma, size=n)` | $\mu$ | $\sigma^2$ |
| Binomial, $k$ trials | `rng.binomial(n=k, p=p, size=n)` | $kp$ | $kp(1-p)$ |
| Poisson | `rng.poisson(lam=lam, size=n)` | $\lambda$ | $\lambda$ |
| Exponential | `rng.exponential(scale=beta, size=n)` | $\beta$ | $\beta^2$ |

For a Poisson process with rate $r$ events per hour, the expected count in $t$
hours is $\lambda=rt$. The mean waiting time is $\beta=1/r$ **hours**.
`exponential` takes this scale, not the rate.

The probability laws make these parameterizations explicit:

$$
P(K=k)=\binom{N}{k}p^k(1-p)^{N-k},\qquad
P(C=k)=e^{-\lambda}\frac{\lambda^k}{k!},
$$

$$
f_{\text{exp}}(t)=\frac1\beta e^{-t/\beta},\quad t\geq0.
$$

For the normal distribution, `scale` is the standard deviation, not its square.
To sample variance $4$, pass `scale=2`.

```python
import numpy as np

rng = np.random.default_rng(42)
rate_per_hour = 6.0
normal = rng.normal(loc=10, scale=2, size=1000)
successes = rng.binomial(n=10, p=0.3, size=1000)
counts = rng.poisson(lam=rate_per_hour * 0.5, size=1000)
waiting_hours = rng.exponential(scale=1 / rate_per_hour, size=1000)

print(normal.shape, successes.shape, counts.shape, waiting_hours.shape)
print(bool(np.all((successes >= 0) & (successes <= 10))))
print(bool(np.all(counts >= 0) and np.all(waiting_hours >= 0)))
print(f"normal model variance: {2**2}")
print(f"expected count in 30 minutes: {rate_per_hour * 0.5:.1f}")
print(f"mean waiting time in minutes: {60 / rate_per_hour:.1f}")
```

```text
(1000,) (1000,) (1000,) (1000,)
True
True
normal model variance: 4
expected count in 30 minutes: 3.0
mean waiting time in minutes: 10.0
```

`size=1000` requests a thousand draws; it does not turn a rate into the expected
count for a thousand time units. Set the distribution parameter for the intended
exposure separately.

## Monte Carlo connects geometry, probability, and sampling error

Draw $(U_i,V_i)$ uniformly from the unit square. The event $U_i^2+V_i^2\leq1$
covers a quarter circle, so its probability is $p=\pi/4$.
With indicator $I_i$ for this event,

$$
\widehat\pi=4\bar I,\qquad
\operatorname{SE}(\widehat\pi)=4\sqrt{\frac{p(1-p)}N}.
$$

At $N=100{,}000$, the theoretical standard error is about $0.0052$.

```python
import numpy as np

rng = np.random.default_rng(42)
N = 100_000
points = rng.random((N, 2))
inside = np.sum(points**2, axis=1) <= 1.0
estimate = 4 * inside.mean()
p = np.pi / 4
standard_error = 4 * np.sqrt(p * (1 - p) / N)

print(f"pi estimate: {estimate:.3f}")
print(f"theoretical standard error: {standard_error:.4f}")
print(f"points for half the standard error: {4 * N}")
```

```text
pi estimate: 3.150
theoretical standard error: 0.0052
points for half the standard error: 400000
```

The estimate is a random result, not an exact area computation. The displayed
seeded estimate is rounded; exact random outputs can change with the environment.
The $N^{-1/2}$ scaling means halving typical sampling error requires four times
as many points, not twice as many. Increasing $N$ does not guarantee that every
individual run moves closer to $\pi$.

The formula above uses the known probability to demonstrate the error scale.
For an unknown Bernoulli probability, an estimated standard error substitutes
the observed fraction; normal-approximation intervals require care for small
samples or fractions near zero or one.

See NumPy's [variance and standard deviation conventions](https://numpy.org/doc/stable/reference/generated/numpy.std.html),
[quantile methods](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html),
[covariance orientation](https://numpy.org/doc/stable/reference/generated/numpy.cov.html), and
[random generators](https://numpy.org/doc/stable/reference/random/generator.html).

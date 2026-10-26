# Searching, filtering, and sorting without losing meaning

A search may need values, coordinates, or an ordering. These are different
outputs. Decide which one you need before choosing `where`, `nonzero`,
`argsort`, or `searchsorted`.

For a predicate $P$ on entries, a boolean mask stores
$M_i=\mathbf 1\{P(x_i)\}$. The mask has the same shape as the entries tested.

## Values and coordinates are not interchangeable

```python
import numpy as np

A = np.array([[1, 9, 2], [8, 3, 7]])
mask = A > 6
row, col = np.nonzero(mask)

print(mask.tolist())
print(A[mask].tolist())
print(row.tolist(), col.tolist())
print(np.argwhere(mask).tolist())
print(A[row, col].tolist())
print(np.where(mask, A, 0).tolist())
```

```text
[[False, True, False], [True, False, True]]
[9, 8, 7]
[0, 1, 1] [1, 0, 2]
[[0, 1], [1, 0], [1, 2]]
[9, 8, 7]
[[0, 9, 0], [8, 0, 7]]
```

A full-shape mask `A[mask]` gathers the matching entries into a one-dimensional
copy, in C index order. `np.nonzero` returns a tuple of coordinate arrays suitable
for indexing. `np.argwhere` returns one row of coordinates per match; it is useful
as a coordinate table, but `A[np.argwhere(mask)]` is not equivalent to `A[mask]`.

`np.where(mask)` is the index-returning form. With three arguments,
`np.where(mask, A, 0)` selects values and preserves the broadcast result shape.

## Combine array predicates with parentheses and elementwise logic

Use `&`, `|`, and `~` for elementwise AND, OR, and NOT on boolean arrays.
Parenthesize comparisons: `(x > 1) & (x < 5)`.
Python's `and` and `or` ask for a single truth value and cannot combine
multi-element arrays this way.

```python
import numpy as np

x = np.array([0, 1, 2, 3, 4, 5, 6])
mask = (x > 1) & (x < 5) & (x % 2 == 0)
print(x[mask].tolist())
print(x[~mask].tolist())

try:
    (x > 1) and (x < 5)
except ValueError:
    print("and cannot combine these arrays")
```

```text
[2, 4]
[0, 1, 3, 5, 6]
and cannot combine these arrays
```

Use `np.any(mask)` or `np.all(mask)` when you actually want to reduce a mask to
one boolean. For a matrix, specify `axis` to express “any feature in this row”
or “all observations in this column.”

## `where` does not make its arguments lazy

The intended expression $y_i=1/x_i$ for $x_i\neq0$, and $y_i=0$ otherwise,
looks like `np.where(x != 0, 1 / x, 0)`.
But Python evaluates `1 / x` **before** `where` is called, including at zero.
The selection cannot prevent that invalid intermediate.

```python
import numpy as np

x = np.array([2.0, 0.0, -4.0])
with np.errstate(divide="raise", invalid="raise"):
    try:
        np.where(x != 0, 1.0 / x, 0.0)
    except FloatingPointError:
        print("division evaluated at zero")

safe = np.divide(1.0, x, out=np.zeros_like(x), where=x != 0)
print(safe.tolist())
```

```text
division evaluated at zero
[0.5, 0.0, -0.25]
```

The ufunc's own `where=` mask skips the excluded divisions.
Initialize `out`: values at false mask positions retain the destination contents.
Forgetting `out` can leave those positions uninitialized.

## Missing, infinite, and approximately equal are different conditions

NaN is unequal to itself. Test it with `np.isnan`.
If the requirement is finite numeric data, use `np.isfinite`; merely excluding
NaNs would still admit positive and negative infinity.

```python
import numpy as np

x = np.array([1.0, np.nan, np.inf, 3.0])
print(bool(x[1] == np.nan))
print(np.isnan(x).tolist())
print(x[np.isfinite(x)].tolist())
print(bool(np.isclose(1e-9, 0.0)))
print(bool(np.isclose(1e-9, 0.0, rtol=0, atol=1e-12)))
```

```text
False
[False, True, False, False]
[1.0, 3.0]
True
False
```

For finite values, `isclose(a, b)` tests approximately

$$
|a-b|\leq atol+rtol\,|b|.
$$

The absolute tolerance matters near zero; the default accepts $10^{-9}$ as
close to zero. That may be wrong if $10^{-9}$ is a meaningful signal in your
units. The reference magnitude is $|b|$, so the test is not generally symmetric.
Also, `allclose` broadcasts inputs; check shape separately if matching shape is
part of correctness.

## Sorting columns independently can destroy records

Suppose each row is `[record_id, score]`. Sorting along `axis=0` sorts each
column independently; it does not keep a row's fields together.
Compute an ordering from the key, then apply it to whole rows.

```python
import numpy as np

records = np.array([[101, 30], [102, 10], [103, 20], [104, 10]])
order = np.argsort(records[:, 1], kind="stable")

print(np.sort(records, axis=0).tolist())
print(order.tolist())
print(records[order].tolist())
print(records.tolist())

values = np.array([3, 1, 2])
result = values.sort()
print(values.tolist(), result)
```

```text
[[101, 10], [102, 10], [103, 20], [104, 30]]
[1, 3, 2, 0]
[[102, 10], [104, 10], [103, 20], [101, 30]]
[[101, 30], [102, 10], [103, 20], [104, 10]]
[1, 2, 3] None
```

The incorrect first result invents the record `[101,10]`.
A stable ordering keeps record $102$ before $104$ because their scores tie and
that was their original order. `np.sort` returns a sorted copy, whereas
`values.sort()` changes `values` and returns `None`.

`axis=-1` sorts within each row of a two-dimensional array.
`axis=None` sorts all entries as a flattened sequence.
Neither operation means “sort the rows by a chosen column.”

For multiple keys, `np.lexsort` uses its **last** key as the primary key:

```python
import numpy as np

group = np.array(["B", "A", "B", "A"])
score = np.array([2, 2, 1, 1])
order = np.lexsort((score, group))  # group first, then score within each group
print(order.tolist())
print(group[order].tolist())
print(score[order].tolist())
```

```text
[3, 1, 2, 0]
['A', 'A', 'B', 'B']
[1, 2, 1, 2]
```

## Binary search gives insertion positions, not automatic membership

For sorted $a$, `searchsorted(a, v, side="left")` finds the first index whose
value is at least $v$. `side="right"` finds the first index whose value is
greater than $v$. Their difference counts occurrences of $v$.

```python
import numpy as np

a = np.array([10, 20, 20, 40])
left = np.searchsorted(a, 20, side="left")
right = np.searchsorted(a, 20, side="right")
print(int(left), int(right), int(right - left))

query = 25
position = np.searchsorted(a, query)
found = position < a.size and a[position] == query
print(int(position), bool(found))
print(int(np.searchsorted(a, 50)))
```

```text
1 3 2
3 False
4
```

Position $3$ for $25$ is where an insertion would preserve order; it is not a
match. Position $4$ for $50$ is one past the last element.
Check the bounds before indexing, and use an ascending sorted input.

## Top-k selection does not require a full sort

`argpartition` separates entries around an order statistic, but does not order
all entries on either side. Sort the selected subset if ranked output is needed.

```python
import numpy as np

scores = np.array([40, 10, 50, 20, 30])
k = 2  # this pattern assumes 1 <= k <= scores.size
candidate = np.argpartition(scores, -k)[-k:]
ranked = candidate[np.argsort(scores[candidate])[::-1]]

print(ranked.tolist())
print(scores[ranked].tolist())
```

```text
[2, 0]
[50, 40]
```

With ties at the cutoff, `argpartition` does not define which tied indices enter
the subset. Use a full stable ordering and an explicit tie policy when identity
matters. Handle `k=0` separately: `[-0:]` means the entire array, not an empty
selection. Filter or define the treatment of NaNs before selecting “largest.”

## Empty matches expose two useful edge cases

`argmax` on an all-false, nonempty mask returns zero because every entry ties
for the maximum. It does not mean a match was found. Also, `all` on an empty
boolean array is true: there is no counterexample to the universal condition.

```python
import numpy as np

x = np.array([1, 2, 3])
mask = x > 10
selected = x[mask]

print(int(np.argmax(mask)), np.flatnonzero(mask).tolist())
print(selected.shape)
print(bool(np.any(selected > 0)), bool(np.all(selected > 0)))
print(bool(selected.size > 0 and np.all(selected > 0)))
```

```text
0 []
(0,)
False True
False
```

If the claim is “there is at least one selected value, and every selected value
is positive,” test both clauses.

See NumPy's [value selection](https://numpy.org/doc/stable/reference/generated/numpy.where.html),
[stable index sorting](https://numpy.org/doc/stable/reference/generated/numpy.argsort.html),
and [insertion search](https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html) references.

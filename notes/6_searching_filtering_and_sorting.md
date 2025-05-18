## Searching, Filtering and Sorting

NumPy provides a set of functions for searching, filtering, and sorting arrays. These operations are helpful for efficiently managing and preprocessing large datasets, enabling you to extract meaningful information, organize data, and prepare it for further analysis or machine learning tasks. This guide covers the fundamental functions for searching within arrays, filtering elements based on conditions, and sorting arrays, along with practical examples to demonstrate their usage.

### Mathematical Intro

From an abstract viewpoint, **searching, filtering, and sorting** all act on an array $A$ by *transforming* or *re-indexing* its discrete domain.  For a 1-D array $A:\{0,\dots ,n-1\}\!\to\!\mathbb F$ let

$$
I_{!\phi} = \{\,i \mid \phi!\bigl(A_i\bigr) = \mathrm{True}\}
$$

$$
\chi_{!\phi}(i) =
\begin{cases}
1, & i \in I_{!\phi},\\
0, & \text{otherwise}
\end{cases}
$$

where $\phi$ is any Boolean predicate (e.g.\ “$A_i>3$” or “$A_i=v$”).

* **Searching** computes the *index set* $I_{\!\phi}$.
* **Filtering** forms the *sub-vector* $(A_i)_{i\in I_{!\phi}}$ , which in NumPy is written `A[χ_φ==1]`.
* **Sorting** finds a *permutation* $\pi\in S_n$ such that $A_{\pi(0)}\le A_{\pi(1)}\le\cdots\le A_{\pi(n-1)}$; in matrix notation this is $A^{\uparrow}=P_{\pi}A$ with permutation matrix $P_{\pi}$.

```
# 1-D vector: illustrate search & filter for A_i > 3
Index →  0   1   2   3   4   5
Value  [ 2 | 3 | 5 | 4 | 1 | 6 ]
Mask   [ 0   0   1   1   0   1 ]   ← χφ
             ↑   ↑       ↑
Search:  Iφ = {2, 3, 5}
Filter:  A[χφ==1] → [5, 4, 6]
```

```
# Sorting the same vector (ascending)
Unsorted: [ 2 | 3 | 5 | 4 | 1 | 6 ]
           0   1   2   3   4   5   ← original indices
Permutation π = (4,0,1,3,2,5)
Sorted:    [ 1 | 2 | 3 | 4 | 5 | 6 ]
           4   0   1   3   2   5   ← original positions after Pπ
```

```
# 2-D array: search for value 1
        columns →
        0   1   2
row 0 [ 0 | 1 | 7 ]
row 1 [ 1 | 5 | 1 ]
row 2 [ 4 | 1 | 3 ]

Matches at (0,1), (1,0), (1,2), (2,1)
```

### Searching

Searching within arrays involves locating the indices of elements that meet specific criteria or contain particular values. NumPy's `np.where()` function is a powerful tool for this purpose, allowing you to identify the positions of elements that satisfy given conditions.

#### Example with 1D Array

```python
import numpy as np

array = np.array([0, 1, 2, 3, 4, 5])
# Find the index where the value is 2
indices = np.where(array == 2)
print(indices[0])  # Expected: [2]

# Find values greater than 1 and less than 4
selected_values = array[np.where((array > 1) & (array < 4))]
print(selected_values)  # Expected: [2, 3]
```

- `np.where(array == 2)`: This function scans the array and returns the indices where the condition `array == 2` is `True`. In this case, it finds that the value `2` is located at index `2`.
- `np.where((array > 1) & (array < 4))`: This compound condition searches for elements greater than `1` and less than `4`. The `&` operator combines both conditions, and `np.where` returns the indices of elements that satisfy both.
- Searching is useful when you need to locate specific data points within a dataset, such as finding all instances of a particular value or identifying data points that fall within a certain range for further analysis.

#### Example with 2D Array

```python
array_2D = np.array([[0, 1], [1, 1], [5, 9]])
# Find the indices where the value is 1
indices = np.where(array_2D == 1)

for row, col in zip(indices[0], indices[1]):
    print(f"Value 1 found at row {row}, column {col}")  # Expected: Three occurrences
```

- `np.where(array_2D == 1)`: This function searches the 2D array for all elements equal to `1` and returns their row and column indices.
- `zip(indices[0], indices[1])`: Combines the row and column indices into pairs, allowing iteration over each position where the value `1` is found.
- In applications like image processing, searching can help identify specific pixel values or regions of interest within an image matrix.

### Filtering

Filtering involves extracting elements from an array that meet certain conditions. NumPy's boolean indexing enables you to create masks based on conditions and use these masks to filter the array efficiently.

#### Example

```python
array = np.array([0, 1, 2, 3, 4, 5])
# Filter values greater than 1 and less than 4
filtered_array = array[(array > 1) & (array < 4)]
print(filtered_array)  # Expected: [2, 3]
```

- `(array > 1) & (array < 4)`: This creates a boolean mask where each element is `True` if it satisfies both conditions (greater than `1` and less than `4`) and `False` otherwise.
- `array[boolean_mask]`: Applying the boolean mask to the array extracts only the elements where the mask is `True`.
- Filtering is commonly used in data preprocessing to select subsets of data that meet specific criteria, such as selecting all records within a certain age range or filtering out outliers in a dataset.

### Sorting

Sorting arrays arranges the elements in a specified order, either ascending or descending. NumPy's `np.sort()` function sorts the array and returns a new sorted array, leaving the original array unchanged. Sorting is fundamental for organizing data, preparing it for search algorithms, and enhancing the readability of datasets.

#### Example with 1D Array

```python
array = np.array([3, 1, 4, 2, 5])
# Sort the array
sorted_array = np.sort(array)
print(sorted_array)  # Expected: [1, 2, 3, 4, 5]
```

- `np.sort(array)`: This function sorts the elements of the array in ascending order and returns a new sorted array.
- Sorting is useful when preparing data for binary search operations, generating ordered lists for reporting, or organizing data for visualization purposes.

#### Example with 2D Array

```python
array_2D = np.array([[3, 1], [4, 2], [5, 0]])
# Sort the array along the first axis (columns)
sorted_array_2D = np.sort(array_2D, axis=0)
print(sorted_array_2D)
```

Expected output:
```
[[3 0]
 [4 1]
 [5 2]]
```

- `np.sort(array_2D, axis=0)`: The `axis=0` parameter specifies that the sort should be performed along the first axis (i.e., down each column). Each column is sorted independently.
- Sorting along specific axes is useful in scenarios where you need to order data within rows or columns, such as organizing features in a dataset or preparing data matrices for statistical analysis.

### Advanced Examples and Techniques

Beyond basic searching, filtering, and sorting, NumPy offers more advanced techniques to handle complex data manipulation tasks efficiently.

#### Sorting Along Different Axes

Sorting in multi-dimensional arrays can be performed along different axes to achieve varied ordering based on rows or columns.

```python
array_2D = np.array([[3, 1], [4, 2], [5, 0]])
# Sort the array along the second axis (rows)
sorted_array_2D_axis1 = np.sort(array_2D, axis=1)
print("Sorted along axis 1:\n", sorted_array_2D_axis1)
```

Expected output:
```
Sorted along axis 1:
[[1 3]
 [2 4]
 [0 5]]
```

- `np.sort(array_2D, axis=1)`: The `axis=1` parameter specifies that the sort should be performed along the second axis (i.e., across each row). Each row is sorted independently.
- Sorting rows can be useful when each row represents a separate entity, and you need to order elements within each entity, such as sorting scores for different students.

#### Using Argsort

The `np.argsort()` function returns the indices that would sort an array. This is particularly useful for indirect sorting or when you need to sort one array based on the ordering of another.

```python
array = np.array([3, 1, 4, 2, 5])
sorted_indices = np.argsort(array)
print("Sorted indices:\n", sorted_indices)
print("Array sorted using indices:\n", array[sorted_indices])
```

Expected output:
```
Sorted indices:
[1 3 0 2 4]
Array sorted using indices:
[1 2 3 4 5]
```

- `np.argsort(array)`: This function returns an array of indices that would sort the original array. In this case, it indicates that the smallest element `1` is at index `1`, followed by `2` at index `3`, and so on.
- `array[sorted_indices]`: Using the sorted indices to reorder the original array results in a sorted array.
- `argsort` is useful when you need to sort multiple arrays based on the order of one array, such as sorting a list of names based on corresponding scores.

#### Complex Filtering

Combining multiple conditions allows for more sophisticated filtering of array elements, enabling the extraction of subsets that meet all specified criteria.

```python
array = np.array([0, 1, 2, 3, 4, 5])
# Complex condition: values > 1, < 4, and even
complex_filtered_array = array[(array > 1) & (array < 4) & (array % 2 == 0)]
print(complex_filtered_array)  # Expected: [2]
```

- `(array > 1) & (array < 4) & (array % 2 == 0)`: This creates a boolean mask that is `True` only for elements that are greater than `1`, less than `4`, and even.
- `array[boolean_mask]`: Applying the complex boolean mask filters the array to include only elements that satisfy all three conditions.
- Complex filtering is essential in data analysis tasks where multiple criteria must be met simultaneously, such as selecting records within a specific range and meeting a particular category or status.

### Practical Applications

* When you start a project, the first step is to **clean your data**—that means spotting and removing anything that’s clearly wrong or out of place.
* Next, you’ll want to **pick the right features**: look at which variables really move the needle and leave out the rest.
* Sometimes you only need a slice of your dataset—**pull out subsets** that match certain conditions to zoom in on what matters.
* In big collections of data, a smart **search strategy** can save you hours by pinpointing patterns or specific values fast.
* Efficient workflows often rely on **scanning arrays** quickly to find exactly the information you need.
* To make sense of a jumble of numbers, **sorting** puts them in order—chronological, alphabetical, or by size—so you can read them at a glance.
* If you only care about items above a certain threshold (say, sales over $1,000), **filtering** helps you isolate those entries.
* When your dataset balloons to millions of rows, you’ll need **robust search, filter, and sort tools** that keep things speedy.
* Writing your own **array-manipulation routines** can give you extra speed and custom options beyond builtin functions.
* Understanding the difference between a **linear search** (checking one by one) and a **binary search** (splitting the list in half) lets you choose what’s quickest for your data size.
* Simple **range filters** (e.g., keep values between 0 and 100) or more complex **conditional checks** can both be part of your toolbox.
* Popular **sorting methods** like quicksort or mergesort each have their own pros and cons when it comes to speed and memory use.
* By **combining** search, filter, and sort steps in the right order, you can turn a slow, clunky process into a smooth pipeline.
* These array skills aren’t just academic—they’re what you’ll use to **prep data for machine-learning** models, too.
* If you’re working with data over time (stock prices, sensor readings), precise **filtering and ordering** are crucial for any meaningful **time-series analysis**.
* Every millisecond counts at scale, so **tuning your search and sort operations** can drastically cut computational load.
* In fields from finance to IoT, having fast **real-time search and filtering** capabilities is becoming non-negotiable.

### Summary Table

| Operation                  | Method/Function      | Description                                                                                   | Example Code                                                                                           | Example Output                                   |
|----------------------------|----------------------|-----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|-------------------------------------------------|
| **Search (1D)**            | `np.where()`         | Finds indices where conditions are met.                                                       | `np.where(array == 2)`                                                                                  | `[2]`                                            |
| **Search (2D)**            | `np.where()`         | Finds indices in a 2D array where conditions are met.                                         | `np.where(array_2D == 1)`                                                                               | `(array([0, 1, 1]), array([1, 0, 1]))`           |
| **Filter**                 | Boolean Indexing     | Extracts elements that satisfy specific conditions.                                           | `array[(array > 1) & (array < 4)]`                                                                      | `[2, 3]`                                         |
| **Sort (1D)**              | `np.sort()`          | Sorts an array and returns a sorted copy.                                                     | `np.sort(array)`                                                                                        | `[1, 2, 3, 4, 5]`                                |
| **Sort (2D, axis=0)**      | `np.sort(array, axis=0)` | Sorts a 2D array along the specified axis.                                                    | `np.sort(array_2D, axis=0)`                                                                              | `[[3, 0], [4, 1], [5, 2]]`                       |
| **Sort (2D, axis=1)**      | `np.sort(array, axis=1)` | Sorts a 2D array along the specified axis.                                                    | `np.sort(array_2D, axis=1)`                                                                              | `[[1, 3], [2, 4], [0, 5]]`                       |
| **Argsort**                | `np.argsort()`       | Returns indices that would sort an array.                                                     | `np.argsort(array)`                                                                                     | `[1, 3, 0, 2, 4]`                                |
| **Complex Filter**         | Boolean Indexing     | Combines multiple conditions for complex filtering.                                           | `array[(array > 1) & (array < 4) & (array % 2 == 0)]`                                                   | `[2]`                                            |

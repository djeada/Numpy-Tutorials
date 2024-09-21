## Searching, Filtering and Sorting

NumPy offers a suite of functions designed for searching within, filtering, and sorting arrays. These capabilities are indispensable when managing and preprocessing datasets, particularly large ones. This guide will cover the essential functions and provide detailed explanations and practical examples to help you utilize these tools effectively.

### Searching

To locate the indices of specific values or those that satisfy a particular condition within an array, you can utilize `np.where()`.

#### Example with 1D Array

```Python
import numpy as np

array = np.array([0, 1, 2, 3, 4, 5])
# Find the index where the value is 2
indices = np.where(array == 2)
print(indices[0])  # Expected: [2]

# Find values greater than 1 and less than 4
selected_values = array[np.where((array > 1) & (array < 4))]
print(selected_values)  # Expected: [2, 3]
```

Explanation:

- `np.where(array == 2)`: Finds the indices where the value is 2.
- `np.where((array > 1) & (array < 4))`: Finds the indices where values are greater than 1 and less than 4.

#### Example with 2D Array

```Python
array_2D = np.array([[0, 1], [1, 1], [5, 9]])
# Find the indices where the value is 1
indices = np.where(array_2D == 1)

for row, col in zip(indices[0], indices[1]):
    print(array_2D[row, col])  # Expected: 1, 1, 1
```

Explanation:

- `np.where(array_2D == 1)`: Finds the indices where the value is 1 in a 2D array.
- `zip(indices[0], indices[1])`: Pairs the row and column indices to iterate over and access the elements.

### Filtering

Boolean indexing enables the extraction of elements that satisfy specific conditions from an array.

#### Example

```Python
array = np.array([0, 1, 2, 3, 4, 5])
# Filter values greater than 1 and less than 4
filtered_array = array[(array > 1) & (array < 4)]
print(filtered_array)  # Expected: [2, 3]
```

Explanation:

- `(array > 1) & (array < 4)`: Creates a boolean mask where the condition is true.
- `array[boolean_mask]`: Filters the array based on the boolean mask.

### Sorting

For sorting arrays, NumPy offers the `np.sort()` function. It produces a sorted copy of the array while leaving the initial array untouched.

#### Example with 1D Array

```Python
array = np.array([3, 1, 4, 2, 5])
# Sort the array
sorted_array = np.sort(array)
print(sorted_array)  # Expected: [1, 2, 3, 4, 5]
```

Explanation:

- `np.sort(array)`: Returns a sorted copy of the array.

#### Example with 2D Array

When sorting multidimensional arrays, you can specify the sorting axis using the `axis` parameter.

```Python
array_2D = np.array([[3, 1], [4, 2], [5, 0]])
# Sort the array along the first axis (rows)
sorted_array_2D = np.sort(array_2D, axis=0)
print(sorted_array_2D)
```

Expected output:

```
[[3 0]
 [4 1]
 [5 2]]
```

Explanation:

- `np.sort(array_2D, axis=0)`: Sorts the array along the specified axis. Here, it sorts each column independently.

### Advanced Examples and Techniques

#### Sorting Along Different Axes

Sorting along different axes in a 2D array can yield different results based on the chosen axis.

```Python
# Sort the array along the second axis (columns)
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

#### Using Argsort

`np.argsort()` returns the indices that would sort an array, which can be useful for indirect sorting.

```Python
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

#### Complex Filtering

Combining multiple conditions for more complex filtering scenarios.

```Python
array = np.array([0, 1, 2, 3, 4, 5])
# Complex condition: values > 1, < 4, and even
complex_filtered_array = array[(array > 1) & (array < 4) & (array % 2 == 0)]
print(complex_filtered_array)  # Expected: [2]
```

### Practical Applications
Understanding how to search, filter, and sort arrays is crucial for various data manipulation tasks, including:

- **Data cleaning** is essential for filtering out invalid or unwanted data points.
- **Feature selection** involves sorting and selecting important features based on statistical criteria.
- Data analysis benefits greatly from extracting subsets of data that meet specific **conditions** for deeper insights.
- Optimized searching allows for quickly locating **data points** or patterns within large datasets.
- Efficient data processing relies on the ability to search through arrays to find **relevant information**.
- **Sorting arrays** helps in organizing data in a meaningful order, making it easier to read and interpret.
- Filtering arrays enables the **isolation** of specific data points that match given criteria, improving data relevance.
- Large-scale data manipulation often requires robust methods for searching, filtering, and sorting to handle the **volume** effectively.
- Developing **algorithms** for array manipulation can enhance performance in data-intensive applications.
- Understanding different **search algorithms**, such as linear and binary search, helps in selecting the most appropriate method for the task.
- Various **filtering techniques**, including range checks and conditional filtering, can be applied to refine data sets.
- **Sorting algorithms**, such as quicksort and mergesort, play a significant role in data organization and retrieval speed.
- Combining search, filter, and sort functions can streamline complex **data processing** workflows.
- Array manipulation skills are fundamental for implementing **machine learning** models and preparing data for training.
- Advanced data analysis tasks, such as **time series analysis**, often depend on precise filtering and sorting of data.
- Optimizing search and sort operations can significantly reduce **computational costs** and improve application performance.
- Search and filter capabilities are critical for **real-time data processing** and analytics in various industries.

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

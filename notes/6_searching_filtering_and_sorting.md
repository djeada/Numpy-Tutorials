## Searching, Filtering and Sorting

NumPy provides a comprehensive set of functions for searching, filtering, and sorting arrays. These operations are essential for efficiently managing and preprocessing large datasets, enabling you to extract meaningful information, organize data, and prepare it for further analysis or machine learning tasks. This guide covers the fundamental functions for searching within arrays, filtering elements based on conditions, and sorting arrays, along with practical examples to demonstrate their usage.

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

Explanation:

- `np.where(array == 2)`: This function scans the array and returns the indices where the condition `array == 2` is `True`. In this case, it finds that the value `2` is located at index `2`.
- `np.where((array > 1) & (array < 4))`: This compound condition searches for elements greater than `1` and less than `4`. The `&` operator combines both conditions, and `np.where` returns the indices of elements that satisfy both.
- **Practical Use Case:** Searching is useful when you need to locate specific data points within a dataset, such as finding all instances of a particular value or identifying data points that fall within a certain range for further analysis.

#### Example with 2D Array

```python
array_2D = np.array([[0, 1], [1, 1], [5, 9]])
# Find the indices where the value is 1
indices = np.where(array_2D == 1)

for row, col in zip(indices[0], indices[1]):
    print(f"Value 1 found at row {row}, column {col}")  # Expected: Three occurrences
```

Explanation:

- `np.where(array_2D == 1)`: This function searches the 2D array for all elements equal to `1` and returns their row and column indices.
- `zip(indices[0], indices[1])`: Combines the row and column indices into pairs, allowing iteration over each position where the value `1` is found.
- **Practical Use Case:** In applications like image processing, searching can help identify specific pixel values or regions of interest within an image matrix.

### Filtering

Filtering involves extracting elements from an array that meet certain conditions. NumPy's boolean indexing enables you to create masks based on conditions and use these masks to filter the array efficiently.

#### Example

```python
array = np.array([0, 1, 2, 3, 4, 5])
# Filter values greater than 1 and less than 4
filtered_array = array[(array > 1) & (array < 4)]
print(filtered_array)  # Expected: [2, 3]
```

Explanation:

- `(array > 1) & (array < 4)`: This creates a boolean mask where each element is `True` if it satisfies both conditions (greater than `1` and less than `4`) and `False` otherwise.
- `array[boolean_mask]`: Applying the boolean mask to the array extracts only the elements where the mask is `True`.
- **Practical Use Case:** Filtering is commonly used in data preprocessing to select subsets of data that meet specific criteria, such as selecting all records within a certain age range or filtering out outliers in a dataset.

### Sorting

Sorting arrays arranges the elements in a specified order, either ascending or descending. NumPy's `np.sort()` function sorts the array and returns a new sorted array, leaving the original array unchanged. Sorting is fundamental for organizing data, preparing it for search algorithms, and enhancing the readability of datasets.

#### Example with 1D Array

```python
array = np.array([3, 1, 4, 2, 5])
# Sort the array
sorted_array = np.sort(array)
print(sorted_array)  # Expected: [1, 2, 3, 4, 5]
```

Explanation:

- `np.sort(array)`: This function sorts the elements of the array in ascending order and returns a new sorted array.
- **Practical Use Case:** Sorting is useful when preparing data for binary search operations, generating ordered lists for reporting, or organizing data for visualization purposes.

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

Explanation:

- `np.sort(array_2D, axis=0)`: The `axis=0` parameter specifies that the sort should be performed along the first axis (i.e., down each column). Each column is sorted independently.
- **Practical Use Case:** Sorting along specific axes is useful in scenarios where you need to order data within rows or columns, such as organizing features in a dataset or preparing data matrices for statistical analysis.

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

Explanation:

- `np.sort(array_2D, axis=1)`: The `axis=1` parameter specifies that the sort should be performed along the second axis (i.e., across each row). Each row is sorted independently.
- **Practical Use Case:** Sorting rows can be useful when each row represents a separate entity, and you need to order elements within each entity, such as sorting scores for different students.

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

Explanation:

- `np.argsort(array)`: This function returns an array of indices that would sort the original array. In this case, it indicates that the smallest element `1` is at index `1`, followed by `2` at index `3`, and so on.
- `array[sorted_indices]`: Using the sorted indices to reorder the original array results in a sorted array.
- **Practical Use Case:** `argsort` is useful when you need to sort multiple arrays based on the order of one array, such as sorting a list of names based on corresponding scores.

#### Complex Filtering

Combining multiple conditions allows for more sophisticated filtering of array elements, enabling the extraction of subsets that meet all specified criteria.

```python
array = np.array([0, 1, 2, 3, 4, 5])
# Complex condition: values > 1, < 4, and even
complex_filtered_array = array[(array > 1) & (array < 4) & (array % 2 == 0)]
print(complex_filtered_array)  # Expected: [2]
```

Explanation:

- `(array > 1) & (array < 4) & (array % 2 == 0)`: This creates a boolean mask that is `True` only for elements that are greater than `1`, less than `4`, and even.
- `array[boolean_mask]`: Applying the complex boolean mask filters the array to include only elements that satisfy all three conditions.
- **Practical Use Case:** Complex filtering is essential in data analysis tasks where multiple criteria must be met simultaneously, such as selecting records within a specific range and meeting a particular category or status.

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

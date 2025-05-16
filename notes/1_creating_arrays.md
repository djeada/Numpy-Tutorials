## Creating Arrays

NumPy, short for Numerical Python, is an important library for scientific and numerical computing in Python. It introduces the `ndarray`, a powerful multi-dimensional array object that allows for efficient storage and manipulation of large datasets. Unlike standard Python lists, NumPy arrays support vectorized operations, which significantly enhance performance, especially for mathematical computations.

### Mathematical Foundations

Before we create arrays in NumPy, it helps to frame them in the language of mathematics:

| Dimensionality | Mathematical Object | Notation                                                 | Typical Size Symbol |
| -------------- | ------------------- | -------------------------------------------------------- | ------------------- |
| 0-D            | **Scalar**          | $a \in \mathbb R$                                        | —                   |
| 1-D            | **Vector**          | $\mathbf v \in \mathbb R^{n}$                            | $n$                 |
| 2-D            | **Matrix**          | $A \in \mathbb R^{m \times n}$                           | $m, n$              |
| $k$-D          | **Tensor**          | $T \in \mathbb R^{n_1\times n_2\times \dots \times n_k}$ | $n_1,\dots,n_k$     |

*Vectorised code ≈ mathematical notation* — concise, readable, and orders-of-magnitude faster.

**Structure**

* A **vector** stacks numbers in a single direction—think of a point in $n$-dimensional space.
* A **matrix** arranges vectors side-by-side, encoding linear maps such as rotations or projections.
* **Higher-order tensors** extend this idea, capturing multi-way relationships (e.g., RGB images, video, physical simulations).

**Addition & Scalar Multiplication**:

$$
\mathbf u + \mathbf v,\qquad c\,\mathbf v
$$

closed under the usual vector-space axioms.

**Inner/Dot Product (1-D)**:

$$
\langle \mathbf u,\mathbf v\rangle \;=\; \sum_{i=1}^{n}u_i v_i
$$

measuring length and angles.

**Matrix–Vector & Matrix–Matrix Product (2-D)**:

$$
A\mathbf v,\qquad AB
$$

composing linear transformations or solving systems $A\mathbf x=\mathbf b$.

### Creating Arrays from Lists and Tuples

NumPy facilitates the conversion of Python lists and tuples into its own array format seamlessly. This interoperability ensures that you can leverage existing Python data structures while benefiting from NumPy's optimized performance for numerical operations.

#### From a List

```python
import numpy as np

# Creating an array from a list
arr_from_list = np.array([1, 2, 3, 4])
print(arr_from_list)
print(type(arr_from_list))
```

Expected output:

```
[1 2 3 4]
<class 'numpy.ndarray'>
```

- `np.array([1, 2, 3, 4])` takes a Python list as input and converts it into a NumPy array. This transformation enables the use of NumPy's extensive array operations.
- The `print` statements display the array and confirm its type as `numpy.ndarray`, ensuring that the data structure is compatible with NumPy's functions and methods.
- **Practical Use Case:** Converting lists to arrays is common when importing data from sources like CSV files or user inputs, allowing for efficient numerical processing thereafter.

#### From a Tuple

```python
# Creating an array from a tuple
arr_from_tuple = np.array((5, 6, 7, 8))
print(arr_from_tuple)
print(type(arr_from_tuple))
```

Expected output:

```
[5 6 7 8]
<class 'numpy.ndarray'>
```

- `np.array((5, 6, 7, 8))` converts a Python tuple into a NumPy array, preserving the order and elements.
- The `type()` function verifies that the resulting object is a `numpy.ndarray`, ensuring compatibility with NumPy's array-specific operations.
- **Practical Use Case:** Tuples, often used for fixed collections of items, can be efficiently transformed into arrays for scenarios requiring numerical computations, such as statistical analysis or matrix operations.

### Initializing Arrays with Default Values

Initializing arrays with predefined values is a fundamental step in many computational tasks. NumPy offers several functions to create arrays filled with specific default values, providing a solid foundation for further data manipulation and analysis.

#### Array of Zeros

```python
# Initializing an array with zeros
zeros_arr = np.zeros((2, 3))
print(zeros_arr)
```

Expected output:

```
[[0. 0. 0.]
 [0. 0. 0.]]
```

- `np.zeros((2, 3))` creates a 2x3 array where every element is initialized to zero.
- The output displays a two-dimensional array with zeros, represented as floating-point numbers by default.
- **Practical Use Case:** Arrays of zeros are useful when you need to create a placeholder for data that will be updated later, such as initializing weights in a neural network before training.

#### Array of Ones

```python
# Initializing an array with ones
ones_arr = np.ones((2, 3))
print(ones_arr)
```

Expected output:

```
[[1. 1. 1.]
 [1. 1. 1.]]
```

- `np.ones((2, 3))` generates a 2x3 array filled with ones.
- The array elements are displayed as floating-point numbers, each set to one.
- **Practical Use Case:** Arrays of ones can serve as a multiplicative identity in various algorithms, such as setting up initial states in iterative methods or creating masks for data manipulation.

### Generating Arrays with Random Values

Creating arrays populated with random values is essential for simulations, statistical sampling, and initializing parameters in machine learning models. NumPy provides robust functions to generate arrays with different distributions of random numbers.

```python
# Generating an array with random values
random_arr = np.random.rand(2, 3)
print(random_arr)
```

Expected output (values will vary):

```
[[0.5488135  0.71518937 0.60276338]
 [0.54488318 0.4236548  0.64589411]]
```

- `np.random.rand(2, 3)` creates a 2x3 array with random floating-point numbers uniformly distributed between 0 and 1.
- Each execution will produce different values, making it suitable for stochastic processes.
- **Practical Use Case:** Random arrays are crucial in scenarios like Monte Carlo simulations, generating synthetic datasets for testing algorithms, or initializing weights in neural networks to ensure varied starting points for optimization.

### Arrays with Evenly Spaced Values

In many applications, it's necessary to generate arrays with numbers that are evenly spaced within a specific range. NumPy's `linspace` function is designed to create such sequences with precise control over the number of samples and the range.

#### Using `np.linspace()`

```python
# Creating an array with evenly spaced values
evenly_spaced_arr = np.linspace(1, 5, 9)
print(evenly_spaced_arr)
```

Expected output:

```
[1.  1.5 2.  2.5 3.  3.5 4.  4.5 5. ]
```

- `np.linspace(1, 5, 9)` generates 9 evenly spaced numbers starting from 1 and ending at 5, inclusive.
- The function ensures that the spacing between consecutive numbers is uniform, which is useful for various analytical tasks.
- **Practical Use Case:** `linspace` is often used in plotting functions to create smooth curves, in numerical integration for defining intervals, or in generating test data that requires uniform distribution across a range.

### Creating Identity Matrix

An identity matrix is a special type of square matrix where all the elements on the main diagonal are ones, and all other elements are zeros. Identity matrices are fundamental in linear algebra, serving as the multiplicative identity in matrix operations.

#### Using `np.eye()`

```python
# Creating an identity matrix
identity_matrix = np.eye(3)
print(identity_matrix)
```

Expected output:

```
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
```

- `np.eye(3)` constructs a 3x3 identity matrix with ones on the diagonal and zeros elsewhere.
- The function is versatile, allowing for the creation of identity matrices of any specified size.
- **Practical Use Case:** Identity matrices are essential in solving systems of linear equations, performing matrix inversions, and serving as the starting point for iterative algorithms in computer graphics and engineering simulations.

### Creating Arrays with Specific Sequences

Generating arrays with specific numerical sequences is a common requirement in programming, especially when dealing with iterations, indexing, or setting up test cases. NumPy's `arange` function provides a straightforward method to create such sequences with defined start, stop, and step values.

#### Using `np.arange()`

```python
# Creating an array with a specific sequence
sequence_arr = np.arange(0, 10, 2)
print(sequence_arr)
```

Expected output:

```
[0 2 4 6 8]
```

- `np.arange(0, 10, 2)` generates an array starting at 0, up to but not including 10, with a step size of 2.
- The function is similar to Python's built-in `range` but returns a NumPy array instead of a list, enabling further numerical operations.
- **Practical Use Case:** `arange` is useful for creating index arrays for looping, generating specific intervals for plotting, or defining ranges for data slicing and dicing in analysis tasks.

### Summary Table

| Method                | Function           | Description (incl. shape)                            | Example Code             | Example Output                         |
| --------------------- | ------------------ | ---------------------------------------------------- | ------------------------ | -------------------------------------- |
| **From List**         | `np.array()`       | Converts a Python list to a 1-D array, shape `(4,)`  | `np.array([1, 2, 3, 4])` | `[1 2 3 4]`                            |
| **From Tuple**        | `np.array()`       | Converts a Python tuple to a 1-D array, shape `(4,)` | `np.array((5, 6, 7, 8))` | `[5 6 7 8]`                            |
| **Array of Zeros**    | `np.zeros()`       | Initializes an array of zeros, shape `(2, 3)`        | `np.zeros((2, 3))`       | `[[0. 0. 0.] [0. 0. 0.]]`              |
| **Array of Ones**     | `np.ones()`        | Initializes an array of ones, shape `(2, 3)`         | `np.ones((2, 3))`        | `[[1. 1. 1.] [1. 1. 1.]]`              |
| **Random Values**     | `np.random.rand()` | Uniform random floats in \[0, 1), shape `(2, 3)`     | `np.random.rand(2, 3)`   | `[[0.54 0.71 0.60] [0.54 0.42 0.64]]`  |
| **Evenly Spaced**     | `np.linspace()`    | 9 evenly spaced values from 1 to 5, shape `(9,)`     | `np.linspace(1, 5, 9)`   | `[1.  1.5 2.  2.5 3.  3.5 4.  4.5 5.]` |
| **Identity Matrix**   | `np.eye()`         | Identity matrix of order 3, shape `(3, 3)`           | `np.eye(3)`              | `[[1. 0. 0.] [0. 1. 0.] [0. 0. 1.]]`   |
| **Specific Sequence** | `np.arange()`      | Even numbers 0 ≤ n < 10, step 2, shape `(5,)`        | `np.arange(0, 10, 2)`    | `[0 2 4 6 8]`                          |

import numpy as np


def main():

    array = np.array([0, 1, 2, 3, 4, 5])

    # find a key in an array
    key = 2
    condition = array == key
    result = np.where(condition)
    print(result)
    print()

    # max, min and their indices
    print(array.max())
    print()

    print(array.min())
    print()

    indices = array.argmax()
    print(indices)
    print(array[indices])
    print()

    indices = array.argmin()
    print(indices)
    print(array[indices])
    print()

    condition = (array > 1) & (array < 4)
    indices = np.where(condition)
    print(indices)

    matrix = np.array([[0, 1], [9, 1], [5, 9]])

    # find a key in a matrix
    key = 1
    condition = matrix == key
    result = np.transpose(condition.nonzero())
    print(result)
    print()

    # max, min and their indices
    print(matrix.max())
    print()

    print(matrix.min())
    print()

    indices = (matrix == matrix.max()).nonzero()
    print(np.transpose(indices))
    print(matrix[indices])
    print()

    indices = (matrix == matrix.min()).nonzero()
    print(np.transpose(indices))
    print(matrix[indices])
    print()

    unique_values = np.unique(matrix)
    print(unique_values)
    print()

    unique, freq = np.unique(matrix, return_counts=True)
    histogram = dict(zip(unique, freq))
    print(histogram)
    print()

if __name__ == "__main__":
    main()

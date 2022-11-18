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
    
    # sorting
    array = np.array([2, 1, 4, 3, 5])
    result = np.sort(array)
    
    # to srot inplace
    array.sort()
    print(array)
    
    #argsort() returns the indices of the sorted elements
    array = np.array([2, 1, 4, 3, 5])
    i = np.argsort(array)
    print(i)

    rand = np.random.RandomState(42)
    matrix = rand.randint(0, 10, (4, 6))
    print(matrix)
    
    # sort each column of matrix
    np.sort(matrix, axis=0)
    
    # sort each row of matrix
    np.sort(matrix, axis=1)


if __name__ == "__main__":
    main()

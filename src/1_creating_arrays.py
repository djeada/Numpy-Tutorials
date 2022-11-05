import numpy as np


def main():

    # creating an array from 1D list
    list_1D = [0, 9, -3, 5, 4]
    array_a = np.array(list_1D)
    print(array_a)
    print()

    # creating an array from 1D tupple
    tuple_1D = (0, 9, -3, 5, 4)
    array_b = np.array(tuple_1D)
    print(array_b)
    print()

    # soruces are not equal to each other
    # but resulting array are
    print(list_1D == tuple_1D)
    print(array_a == array_b)
    print((array_a == array_b).all())

    # basic info
    print(array_a.shape)
    print(array_a.ndim)
    print(array_a.size)
    print(array_a.dtype)

    # creating an array from 2D list
    list_2D = [[1, 2, 3], [4, 5, 6]]
    matrix = np.array(list_2D)
    print(matrix)
    print()

    # basic info
    print(matrix.shape)
    print(matrix.ndim)
    print(matrix.size)
    print(matrix.dtype)

    list_3D = [
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    ]

    tensor = np.array(list_3D)
    print(tensor)
    print()

    # basic info
    print(tensor.shape)
    print(tensor.ndim)
    print(tensor.size)
    print(tensor.dtype)

    # np.arange(start, end, step) creates an array of ints from start to end with step differece between each element
    array = np.arange(3)
    print(array)
    print()
    array = np.arange(3, 6)
    print(array)
    print()
    array = np.arange(0, 9, 3)
    print(array)
    print()

    # np.linspace(start, end, n) creates an array of n doubles from start to end
    array = np.linspace(0, 10, 5)
    print(array)
    print()

    # np.zeros(shape, dtype=float) creates an array of 0's stored as dtype of specified shape
    array = np.zeros(5)
    print(array)
    print()

    matrix = np.zeros((3, 2))
    print(matrix)
    print()

    # np.ones(shape, dtype=float) creates an array of 1's stored as dtype of specified shape
    array = np.ones(5)
    print(array)
    print()

    matrix = np.ones((3, 2))
    print(matrix)
    print()

    # np.full(n, a) creates an array with number a repeated n times
    array = np.full(7, 2)
    print(array)
    print()

    # np.eye(n, dtype=float) creates an identity matrix of size n x n
    matrix = np.eye(3)
    print(matrix)
    print()

    # np.random.rand(shape) creates an array of random numbers stored as floats of specified shape
    array = np.random.rand(5)
    print(array)
    print()

    matrix = np.random.rand(*(3, 2))
    print(matrix)
    print()

    
if __name__ == "__main__":
    main()

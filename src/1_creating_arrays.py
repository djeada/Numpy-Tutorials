import numpy as np


def main():

    """ 
    1. Creating np array from Python data structures: list and tuple
    - list is not equal to tuple with identical elements
    - arrays created from them are equal
    - basic operations (ex. multiplication) have different meaning for python ds and np arrays
    - you can insert and delete items in a list by assigning a sequence of different length to a slice, 
    whereas np would raise an error.
    """

    list_1D = [0, 9, -3, 5, 4]
    tuple_1D = (0, 9, -3, 5, 4)
    array_from_list_1D = np.array(list_1D)
    array_from_tupple_1D = np.array(tuple_1D)

    assert list_1D != tuple_1D
    assert (array_from_list_1D == array_from_tupple_1D).all()
    assert list_1D * 3 == [0, 9, -3, 5, 4, 0, 9, -3, 5, 4, 0, 9, -3, 5, 4]
    assert (array_from_list_1D * 3 == np.array([0, 27, -9, 15, 12])).all()

    assert array_from_list_1D.shape == (5,)
    assert array_from_list_1D.ndim == 1
    assert array_from_list_1D.size == 5
    assert array_from_list_1D.dtype == "int64"

    list_2D = [[1, 2, 3], [4, 5, 6]]

    array_from_list_2D = np.array(list_2D)

    assert array_from_list_2D.shape == (2, 3)
    assert array_from_list_2D.ndim == 2
    assert array_from_list_2D.size == 6
    assert array_from_list_2D.dtype == "int64"

    list_3D = [
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    ]

    array_from_list_3D = np.array(list_3D)

    assert array_from_list_3D.shape == (3, 3, 3)
    assert array_from_list_3D.ndim == 3
    assert array_from_list_3D.size == 27
    assert array_from_list_3D.dtype == "int64"

    """
    2. Creating np array from numpy methods
    - np.arange(start, end, step) creates an array of ints from start to end with step differece between each element
    - np.linspace(start, end, n) creates an array of doubles from start to end with step differece between each element
    - np.zeros(shape, dtype=float) creates an array of 0's stored as dtype of specified shape
    - np.ones(shape, dtype=float) creates an array of 1's stored as dtype of specified shape
    - np.eye(n, dtype=float) creates an identity matrix of size n x n
    - np.random.rand(shape) creates an array of 1's stored as floats of specified shape
    """

    assert (np.arange(3) == np.array([0, 1, 2])).all()
    assert (np.arange(3, 6) == np.array([3, 4, 5])).all()
    assert (np.arange(0, 9, 3) == np.array([0, 3, 6])).all()

    assert (np.linspace(0, 10, 5) == [0.0, 2.5, 5.0, 7.5, 10.0]).all()

    assert (np.zeros(5) == np.array([0.0, 0.0, 0.0, 0.0, 0.0])).all()
    assert (np.zeros((3, 2)) == np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])).all()

    assert (np.ones(5) == np.array([1.0, 1.0, 1.0, 1.0, 1.0])).all()
    assert (np.ones((3, 2)) == np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])).all()

    assert (
        np.eye(3) == np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    ).all()


if __name__ == "__main__":
    main()

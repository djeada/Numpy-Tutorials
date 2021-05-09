import numpy as np


def main():

    """ 
    There are several methods for regrouping values in arrays.:
    np.reshape(array, shape) without changing the values, changes the shape of an array.
    np.transpose(array) permuting all axes. For 2D case exchanging rows and columns.
    np.swapaxes(array, axis1, axis2) swap any two axes.
    np.rollaxis(array, axis) "rotate" the axes.
    np.fliplr(array) flip array in the left/right direction.
    np.flipud(array) flip array in the up/down direction.
    np.roll(array, n) shift all elements by n positios.
    np.rot90(array) rotate array 90 degrees.
    np.sort(array) sorts the array.
    """

    matrix_A = np.array([[3, 9], [21, -3]])

    assert (np.reshape(matrix_A, (4,)) == np.array([3, 9, 21, -3])).all()
    assert (np.reshape(matrix_A, (1, 4)) == np.array([[3, 9, 21, -3]])).all()
    assert (np.reshape(matrix_A, (4, 1)) == np.array([[3], [9], [21], [-3]])).all()
    assert (np.reshape(matrix_A, (1, 4, 1)) == np.array([[[3], [9], [21], [-3]]])).all()

    assert (np.transpose(matrix_A) == np.array([[3, 21], [9, -3]])).all()

    tensor_A = np.array([[[1, 0], [2, -1]], [[-3, 2], [7, 5]]])

    assert (np.reshape(matrix_A, (4,)) == np.array([3, 9, 21, -3])).all()
    assert (np.reshape(matrix_A, (4,)) == np.array([3, 9, 21, -3])).all()
    assert (np.reshape(matrix_A, (4,)) == np.array([3, 9, 21, -3])).all()

    assert (
        np.swapaxes(tensor_A, 0, 1) == np.array([[[1, 0], [-3, 2]], [[2, -1], [7, 5]]])
    ).all()

    assert (
        np.swapaxes(tensor_A, 0, 2) == np.array([[[1, -3], [2, 7]], [[0, 2], [-1, 5]]])
    ).all()

    assert (
        np.swapaxes(tensor_A, 1, 2) == np.array([[[1, 2], [0, -1]], [[-3, 7], [2, 5]]])
    ).all()

    assert (
        np.rollaxis(tensor_A, 1) == np.array([[[1, 0], [-3, 2]], [[2, -1], [7, 5]]])
    ).all()

    assert (
        np.rollaxis(tensor_A, 2) == np.array([[[1, 2], [-3, 7]], [[0, -1], [2, 5]]])
    ).all()

    assert (np.fliplr(matrix_A) == np.array([[9, 3], [-3, 21]])).all()
    assert (np.flipud(matrix_A) == np.array([[21, -3], [3, 9]])).all()

    assert (np.roll(matrix_A, -1) == np.array([[9, 21], [-3, 3]])).all()
    assert (np.roll(matrix_A, 1) == np.array([[-3, 3], [9, 21]])).all()

    assert (np.rot90(matrix_A) == np.array([[9, -3], [3, 21]])).all()

    array = np.array([8, 28, 17, -8, 0, 11, -38, -92, 1])
    assert (np.sort(array) == np.array([[-92, -38, -8, 0, 1, 8, 11, 17, 28]])).all()


if __name__ == "__main__":
    main()

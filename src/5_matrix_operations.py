import numpy as np


def main():

    """ 
    Basic matrix operations are supported by numpy.
    Addition, subtraction, multiplication, division by:
    - scalar
    - vector 
    - matrix
    """

    matrix_A = np.array([[3, 9], [21, -3]])

    assert (matrix_A + 5 == np.array([[8, 14], [26, 2]])).all()
    assert (matrix_A - 1 == np.array([[2, 8], [20, -4]])).all()
    assert (matrix_A * 2 == np.array([[6, 18], [42, -6]])).all()
    assert (matrix_A / 3 == np.array([[1, 3], [7, -1]])).all()

    vector = [1, 2]
    assert (matrix_A + vector == np.array([[4, 11], [22, -1]])).all()
    assert (matrix_A - vector == np.array([[2, 7], [20, -5]])).all()
    assert (matrix_A * vector == np.array([[3, 18], [21, -6]])).all()
    assert (matrix_A / vector == np.array([[3, 4.5], [21, -1.5]])).all()

    matrix_B = np.array([[4, 9], [25, 16]])

    assert (np.sqrt(matrix_B) == np.array([[2, 3], [5, 4]])).all()

    assert (matrix_A + matrix_B == np.array([[7, 18], [46, 13]])).all()
    assert (matrix_A - matrix_B == np.array([[-1, 0], [-4, -19]])).all()
    assert (matrix_A * matrix_B == np.array([[12, 81], [525, -48]])).all()
    assert (matrix_B / matrix_A == np.array([[4 / 3, 1], [25 / 21, -16 / 3]])).all()


if __name__ == "__main__":
    main()

import numpy as np

# from numpy.linalg import det, inv


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

    """
    The transpose of a matrix is a reversal of its rows with its columns.
    Square matrices have determinants.
    The identity matrix is a square matrix with ones on the diagonal and zeros elsewhere. 
    The inverse of a square matrix M is a matrix of the same size, N, such that M⋅N=I.
    """
    M = np.array([[-4, 5], [1, 7], [8, 3]])
    N = np.array([[3, -5, 2, 7], [-5, 1, -4, -3]])
    print(np.dot(M, N))

    assert (np.transpose(matrix_A) == np.array([[3, 21], [9, -3]])).all()
    assert (np.transpose(matrix_B) == np.array([[4, 25], [9, 16]])).all()

    assert np.linalg.det(matrix_A) == -198
    assert abs(np.linalg.det(matrix_B) + 161) < 0.001

    identity = np.eye(2)
    inverse = np.linalg.inv(matrix_A)
    product = np.dot(matrix_A, inverse)
    assert (product == identity).all()


if __name__ == "__main__":
    main()

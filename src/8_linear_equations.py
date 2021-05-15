import numpy as np
from numpy.linalg import solve


def main():

    """ 
    You can solve linear equations from definition.
    y = A*x
    x = A^-1*y
    or using numpy solve method.
    """

    matrix = np.matrix([[2, 0, 5], [3, 4, 8], [2, 7, 3]])
    inverse = matrix.I

    y = np.matrix([[10], [15], [5]])

    solution = inverse * y

    assert np.allclose(matrix * solution - y, np.array([[0, 0, 0]]))
    assert np.allclose(solution, solve(matrix, y))

    # Eigenvalues and Eigenvectors
    A = np.array([[1, 2], [2, 1]])
    x = np.array([1, 0])

    for i in range(10):
        x = A @ x


if __name__ == "__main__":
    main()

import numpy as np
import math


def main():

    """
    Operations:
    - vector addition
    - scalar multiplication
    - dot product
    - the cross product
    - the angle between two vectors:
    """

    vector_row = np.array([[1, 2, 3, 4]])
    vector_column = np.array([[1], [2], [3], [4]])

    assert vector_row.shape == (1, 4)
    assert vector_column.shape == (4, 1)

    assert (vector_row.T == vector_column).all()
    assert (vector_column.T == vector_row).all()

    assert np.linalg.norm(vector_row, 1) == 4.0
    assert math.isclose(np.linalg.norm(vector_row, 2), 5.5, rel_tol=1e-2)
    assert np.linalg.norm(vector_row, np.inf) == 10.0

    v = np.array([[9, 2, 5]])
    w = np.array([[-3, 8, 2]])
    theta = np.arccos(np.dot(v, w.T) / (np.linalg.norm(v) * np.linalg.norm(w)))

    assert math.isclose(theta[0][0], 1.582, rel_tol=1e-2)
    assert (np.cross(v, w) == [[-36, -33, 78]]).all()


if __name__ == "__main__":
    main()

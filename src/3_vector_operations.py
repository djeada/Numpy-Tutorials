import numpy as np
import math


def main():

    vector_row = np.array([[1, 2, 3, 4]])
    print(vector_row)
    print(vector_row.shape)
    print()

    vector_column = np.array([[1], [2], [3], [4]])
    print(vector_column)
    print(vector_column.shape)
    print()

    vector_row_transposed = vector_row.T
    print(vector_row_transposed)
    print()

    vector_column_transposed = vector_column.T
    print(vector_column_transposed)
    print()

    # norms
    first_norm = np.linalg.norm(vector_row, 1)
    print(first_norm)

    second_norm = np.linalg.norm(vector_row, 2)
    print(second_norm)

    inf_norm = np.linalg.norm(vector_row, np.inf)
    print(inf_norm)

    # addition of vectors
    vector_sum = vector_row + vector_row
    print(vector_sum)
    print()

    scalar_sum = vector_row + 5
    print(scalar_sum)
    print()

    # scalar multiplication
    scalar_multiplication = 3 * vector_row
    print(scalar_multiplication)
    print()

    # dot product
    dot_product = np.dot(vector_row, vector_row.T)
    print(dot_product)
    print()

    v = np.array([[9, 2, 5]])
    w = np.array([[-3, 8, 2]])

    # cross product
    cross_product = np.cross(v, w)
    print(cross_product)
    print()

    # the angle between two vectors
    theta = np.arccos(np.dot(v, w.T) / (np.linalg.norm(v) * np.linalg.norm(w)))
    print(theta)
    print()


if __name__ == "__main__":
    main()

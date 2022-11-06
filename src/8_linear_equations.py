import numpy as np
from numpy.linalg import solve


def main():

    matrix = np.matrix([[2, 0, 5], [3, 4, 8], [2, 7, 3]])
    inverse = matrix.I

    y = np.matrix([[10], [15], [5]])

    solution = inverse * y
    print(solution)

if __name__ == "__main__":
    main()

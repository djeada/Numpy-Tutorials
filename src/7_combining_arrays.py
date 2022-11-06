import numpy as np


def main():
    
    array_a = np.array([0, 1, 2])
    array_b = np.array([4, 5, 6])

    result = np.concatenate((array_a, array_b))
    print(result)
    print()

    result = np.vstack((array_a, array_b))
    print(result)
    print()

    result = np.hstack((array_a, array_b))
    print(result)
    print()

    connected = np.concatenate((array_a, array_b))
    result = np.split(connected, 3)

    for array in result:
        print(array)
        print()

    matrix_a = np.array([[0, 1], [2, 3]])
    matrix_b = np.array([[4, 5], [6, 4]])

    result = np.concatenate((matrix_a, matrix_b))
    print(result)
    print()

    result = np.vstack((matrix_a, matrix_b))
    print(result)
    print()

    result = np.hstack((matrix_a, matrix_b))
    print(result)
    print()

    connected = np.concatenate((matrix_a, matrix_b))
    result = np.split(connected, 4)

    for array in result:
        print(array)
        print()


if __name__ == "__main__":
    main()

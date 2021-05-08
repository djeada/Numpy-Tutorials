import numpy as np


def max_value(array):
    return array.max()


def min_value(array):
    return array.min()


def max_index(array):
    return array.argmax()


def min_index(array):
    return array.argmin()


def main():

    """ 
    - Find index of a given value.
    - Find max and min.
    - Unique elements.
    - Histogram.
    """

    array_1 = np.array([0, 1, 2, 3, 4, 5])

    assert array_1[0] == 0
    assert array_1[-1] == 5
    assert (array_1[1:3] == np.array([1, 2])).all()

    array_1 = np.append(array_1, 6)
    array_1 = np.insert(array_1, 1, 100)

    assert array_1[-1] == 6
    assert array_1[1] == 100

    array_1[1] = 20

    assert array_1[1] == 20

    array_1[:5] += 1

    assert (array_1 == np.array([1, 21, 2, 3, 4, 4, 5, 6])).all()

    array_1 = array_1 / 2

    assert (array_1 == np.array([0.5, 10.5, 1.0, 1.5, 2.0, 2.0, 2.5, 3.0])).all()

    array_1 = array_1 % 3

    assert (array_1 == np.array([0.5, 1.5, 1.0, 1.5, 2.0, 2.0, 2.5, 0.0])).all()

    array_1 = np.delete(array_1, (1, 3))

    assert (array_1 == np.array([0.5, 1.0, 2.0, 2.0, 2.5, 0.0])).all()


if __name__ == "__main__":
    main()

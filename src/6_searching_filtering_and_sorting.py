import numpy as np


def main():

    """
    - Find index of a given value.
    - Find indices of all values meeting given condition.
    - Find max and min.
    - Unique elements.
    - Histogram.
    """

    array = np.array([0, 1, 2, 3, 4, 5])
    number = 2
    result = np.where(array == 2)

    assert array[result[0]] == number
    assert array.max() == 5
    assert array.min() == 0
    assert array[array.argmax()] == 5
    assert array[array.argmin()] == 0

    for i in np.where((array > 1) & (array < 4))[0]:
        num = array[i]
        assert num > 1 and num < 4

    array_2D = np.array([[0, 1], [1, 1], [5, 9]])
    number = 1
    result = np.where(array_2D == number)

    for cor in list(zip(result[0], result[1])):
        assert array_2D[cor[0]][cor[1]] == number

    assert array_2D.max() == 9
    assert array_2D.min() == 0

    unique_values = np.unique(array_2D)

    assert (unique_values == [0, 1, 5, 9]).all()

    unique, freq = np.unique(array_2D, return_counts=True)
    histogram = dict(zip(unique, freq))
    expected_result = {0: 1, 1: 3, 5: 1, 9: 1}

    assert histogram == expected_result


if __name__ == "__main__":
    main()

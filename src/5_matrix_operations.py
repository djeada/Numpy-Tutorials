import numpy as np


def increase_by_k(array, k):
    return array + k


def decrease_by_k(array, k):
    return array - k


def multiply_by_k(array, k):
    return array * k


def divide_by_k(array, k):
    return array / k


def elements_greater_than_k(array, k):
    return array > k


def elements_less_than_k(array, k):
    return array < k


def sqrt_of_each_element(array):
    return np.sqrt(array)


def sum_two(arr_1, arr_2):
    return arr_1 + arr_2


def difference_two(arr_1, arr_2):
    return arr_1 - arr_2


# my_3D_array / my_vector

# my_3D_array % my_vector


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

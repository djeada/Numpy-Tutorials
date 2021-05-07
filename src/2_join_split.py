import numpy as np


def main():

    """ 
    Common functions used to create arrays from existing arrays:
    - np.concatenate((arr1, arr2, ...), axis=0) connects arrays along axis chosen from existing axis
    - np.stack((arr1, arr2, ...), axis=0) connects arrays along new axis (all inputs must have the same shape)
    - np.split(arr, indices_or_sections, axis=0) splits into sub-arrays along chosen axis
    """

    array_1 = np.array([0, 1, 2, 3])
    array_2 = np.array([4, 5, 6])

    result = np.array([0, 1, 2, 3, 4, 5, 6])
    assert (np.concatenate((array_1, array_2)) == result).all()

    array_1 = np.array([[0, 1], [2, 3]])
    array_2 = np.array([[4, 5], [6, 4]])

    result = [[0, 1], [2, 3], [4, 5], [6, 4]]
    assert (np.concatenate((array_1, array_2)) == result).all()

    result = [[0, 1, 4, 5], [2, 3, 6, 4]]
    assert (np.concatenate((array_1, array_2), axis=1) == result).all()

    array_1 = np.array([0, 1, 2])
    array_2 = np.array([3, 4, 5])

    result = np.array([[0, 1, 2], [3, 4, 5]])
    assert (np.stack((array_1, array_2)) == result).all()

    result = np.array([[0, 3], [1, 4], [2, 5]])
    assert (np.stack((array_1, array_2), axis=1) == result).all()

    array_1 = np.array([[0, 1], [2, 3]])
    array_2 = np.array([[4, 5], [6, 4]])

    result = [[[0, 1], [2, 3]], [[4, 5], [6, 4]]]
    assert (np.stack((array_1, array_2)) == result).all()

    result = [[[0, 1], [4, 5]], [[2, 3], [6, 4]]]
    assert (np.stack((array_1, array_2), axis=1) == result).all()

    result = [[[0, 4], [1, 5]], [[2, 6], [3, 4]]]
    assert (np.stack((array_1, array_2), axis=2) == result).all()

    array_1 = np.array([0, 1, 2])
    array_2 = np.array([3, 4, 5])

    connected = np.concatenate((array_1, array_2))
    splited = np.split(connected, 2)

    assert (splited[0] == array_1).all()
    assert (splited[1] == array_2).all()


if __name__ == "__main__":
    main()

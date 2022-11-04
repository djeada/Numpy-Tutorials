import numpy as np


# TODO: add slices, also for 2D
def main():

    """
    - Elements can be accessed using indices.
    - You may modify a single element, a group of elements, or all elements in an array at once.
    - A new element/row may be appended or inserted.
    - Elements and rows can be deleted as well.
    - Selecting elements by condition is possible.
    """

    array = np.array([0, 1, 2, 3, 4, 5])

    assert array[0] == 0
    assert array[-1] == 5
    assert (array[1:3] == np.array([1, 2])).all()

    array = np.append(array, 6)
    array = np.insert(array, 1, 100)

    assert array[-1] == 6
    assert array[1] == 100

    array[1] = 20

    assert array[1] == 20

    array[:5] += 1

    assert (array == np.array([1, 21, 2, 3, 4, 4, 5, 6])).all()

    array = array / 2

    assert (array == np.array([0.5, 10.5, 1.0, 1.5, 2.0, 2.0, 2.5, 3.0])).all()

    array = array % 3

    assert (array == np.array([0.5, 1.5, 1.0, 1.5, 2.0, 2.0, 2.5, 0.0])).all()

    array = np.delete(array, (1, 3))

    assert (array == np.array([0.5, 1.0, 2.0, 2.0, 2.5, 0.0])).all()


if __name__ == "__main__":
    main()

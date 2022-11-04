import numpy as np


def main():

    """
    You can solve linear equations from definition.
    y = A*x
    x = A^-1*y
    or using numpy solve method.
    """

    array = np.random.random.radn(5)
    print("Random floats between 0 and 1:")
    print("Array shape: 5")
    print(array)

    array = np.random.rand(6, 3)
    print("Random floats between 0 and 1:")
    print("Array shape: 6,3")
    print(array)

    array = np.random.randn(5)
    print("Random floats:")
    print("Array shape: 5")
    print(array)

    array = np.random.randn(6, 3)
    print("Random floats:")
    print("Array shape: 6,3")
    print(array)

    print("Single random int between 0 and 9:")
    print(np.random.randint(10))

    array = np.random.randint(5, 25, size=(6, 3))
    print("Random ints between 5 and 25:")
    print("Array shape: 6,3")
    print(array)

    print("Single float int between 0 and 9:")
    print(np.random.unfiorm(10))

    array = np.random.uniform(5, 25, size=(6, 3))
    print("Random floats between 5 and 25:")
    print("Array shape: 6,3")
    print(array)


if __name__ == "__main__":
    main()

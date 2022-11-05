import numpy as np


def main():

    array = np.array([0, 1, 2, 3, 4, 5])
    print(f"element at index 0 in array {array} is {array[0]}")
    print(f"element at the last index in array {array} is {array[-1]}")
    print(f"elements between indices 1 and 3 in array {array} are {array[1:4]}")

    # use append to add an element at the end of array
    array = np.append(array, 6)
    print(array)
    print()

    # use insert to insert an element at a given index
    array = np.insert(array, 3, 8)
    print(array)
    print()

    # modify a single element
    array[2] = 7
    print(array)
    print()

    # modify a number of consecutive elements
    array[1:4] = [-1, -1, -1]
    print(array)
    print()

    # apply an operation to all the elements
    array += 5
    array //= 2
    print(array)
    print()

    # apply a function to all elements
    array = np.sqrt(array)
    print(array)
    print()

    array = np.cos(array)
    print(array)
    print()

    # remove an element
    array = np.delete(array, (1, 3))
    print(array)
    print()

if __name__ == "__main__":
    main()

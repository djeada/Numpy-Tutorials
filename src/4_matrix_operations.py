import numpy as np


def main():

    # addition, subtraction, multiplication, division by scalar
    matrix = np.array([[3, 9], [21, -3]])

    sum_result = matrix + 5
    print(sum_result)
    print()

    diff_result = matrix - 1
    print(diff_result)
    print()

    mult_result = matrix * 2
    print(mult_result)
    print()

    div_result = matrix / 3
    print(div_result)
    print()

    # addition, subtraction, multiplication, division by vector
    vector = np.array([1, 2])

    sum_result = matrix + vector
    print(sum_result)
    print()

    diff_result = matrix - vector
    print(diff_result)
    print()

    mult_result = matrix * vector
    print(mult_result)
    print()

    div_result = matrix / vector
    print(div_result)
    print()

    # addition, subtraction, multiplication, division by matrix
    matrix_b = np.array([[4, 9], [25, 16]])

    sum_result = matrix + matrix_b
    print(sum_result)
    print()

    diff_result = matrix - matrix_b
    print(diff_result)
    print()

    mult_result = matrix * matrix_b
    print(mult_result)
    print()

    div_result = matrix_b / matrix
    print(div_result)
    print()

    # transpose
    transpose_a = np.transpose(matrix)
    print(transpose_a)
    print()

    transpose_b = np.transpose(matrix_b)
    print(transpose_b)
    print()

    # determinant
    determinant = np.linalg.det(matrix)
    print(determinant)
    print()

    # rank
    rank = np.linalg.matrix_rank(matrix)
    print(f"The rank of the matrix: {rank}")
    # inverse
    inverse = np.linalg.inv(matrix)
    print("The inveres of the matrix")
    print(inverse)
    print()

    product = np.dot(matrix, inverse)
    print("The dot product of the matrix and its inverse.")
    print(product.astype(int))
    print()

    # sparsity
    def count_sparsity(matrix):
        temp = np.nan_to_num(matrix, 0)
        sparsity = 1.0 - (np.count_nonzero(temp) / temp.size)
        return sparsity


    matrix = np.array([[1, 1, 0, 1, 0, 0], [1, 0, 2, 0, 0, 1], [99, 0, 0, 2, 0, 0]])

    sparsity = count_sparsity(matrix)
    print(sparsity)

if __name__ == "__main__":
    main()

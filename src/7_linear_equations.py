my_first_matrix = np.matrix([[3, 1, 4], [1, 5, 9], [2, 6, 5]])
my_first_inverse = my_first_matrix.I
right_hand_side = np.matrix([[11], [22], [33]])
solution = my_first_inverse * right_hand_side

my_first_matrix * solution - right_hand_side

from numpy.linalg import solve

solve(my_first_matrix, right_hand_side)

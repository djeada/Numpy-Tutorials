import numpy as np


def main():

  matrix = np.array([[3, 9], [21, -3]])
  print(matrix)
  print()

  reshaped_matrix = np.reshape(matrix, (4,))
  print(reshaped_matrix)
  print()

  reshaped_matrix = np.reshape(matrix, (1, 4))
  print(reshaped_matrix)
  print()
  reshaped_matrix = np.reshape(matrix, (4, 1))
  print(reshaped_matrix)
  print()

  reshaped_matrix = np.reshape(matrix, (1, 4, 1))
  print(reshaped_matrix)
  print()

  result = np.swapaxes(matrix, 0, 1)
  print(result)
  print()

  result = np.swapaxes(matrix, 1, 0)
  print(result)
  print()

  result = np.rollaxis(matrix, 0)
  print(result)
  print()

  result = np.rollaxis(matrix, 1)
  print(result)
  print()

  tensor = np.array([[[1, 0], [2, -1]], [[-3, 2], [7, 5]]])
  print(tensor)
  print()

  result = np.swapaxes(tensor, 0, 1)
  print(result)
  print()

  result = np.swapaxes(tensor, 0, 2)
  print(result)
  print()

  result = np.swapaxes(tensor, 1, 2)
  print(result)
  print()

  result = np.rollaxis(tensor, 1)
  print(result)
  print()

  result = np.rollaxis(tensor, 2)
  print(result)
  print()

  result = np.fliplr(matrix)
  print(result)
  print()

  result = np.flipud(matrix)
  print(result)
  print()

  result = np.roll(matrix, -1)
  print(result)
  print()

  result = np.roll(matrix, 1)
  print(result)
  print()

  result = np.rot90(matrix)
  print(result)
  print()

if __name__ == "__main__":
    main()

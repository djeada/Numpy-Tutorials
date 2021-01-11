import numpy as np

def change_dimension(array, n, m):
	array.reshape(n, m)

def max_value(array):
	array.max()

def min_value(array):
	array.min()

def max_index(array):
	array.argmax()

def min_index(array):
	array.argmin()

def increase_by_k(array, k):
	return array + k

def decrease_by_k(array, k):
	return array - k

def multiply_by_k(array, k):
	return array * k

def divide_by_k(array, k):
	return array / k

def elements_greater_than_k(array, k):
	array > k

def elements_less_than_k(array, k):
	array < k

def sqrt_of_each_element(array):
	return np.sqrt(array)

def sum_two(arr_1, arr_2):
	return arr_1 + arr_2

def difference_two(arr_1, arr_2):
	return arr_1 - arr_2

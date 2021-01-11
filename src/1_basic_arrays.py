import numpy as np

def array_from_list(_list):
	return np.array(_list)

def array_of_ints(start, end, step):
	return np.arange(start, end, step)

def array_of_doubles(start, end, n):
	return np.linspace(start, end, n)

def array_of_zeros(n):
	return np.zeros(n)

def array_of_ones(n):
	return np.ones(n)

def identity_matrix(n):
	return np.eye(n)

def random_doubles(n):
	return np.random.rand(n)

import numpy as np


my_array = np.array([1, 3, 5])

my_matrix = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])

#consecutive integers 
np.arange(0, 21)

#arithmetic sequence
np.arange(0, 21, 3)

#zeros and ones
np.zeros(20)
np.ones(50)

np.linspace(0,100,1000)

#identity matrix
np.eye(10)

#random between 0 and 1
np.random.rand(10)

#change dimension
arr = np.array([0,1,2,3,4,5,6,7,8])
arr.reshape(3,3)

#min and max and indexes
def max_value(arr):
	arr.min()

def min_index(arr):
	arr.argmin()

# Reshape
# Use np.transpose to permute all the axes at once.
# Use np.swapaxes to swap any two axes.
# Use np.rollaxis to "rotate" the axes.
# SORT

# 1-Dimension, no change

# 2-Dimensions,exchange rows and columns (a[i,j] becomes a[j,i])

np.transpose(my_start_array)
np.swapaxes(my_2_3_4_array, 1, 0)
np.rollaxis(my_2_3_4_array, 0, 2)

# Flip array in the left/right direction.
np.fliplr(my_3_8_array)
np.flipud(my_2_3_4_array)
np.roll(my_start_array, 5)
np.rot90(my_3_8_array)
my_3_8_array = my_start_array.reshape((3, 8))

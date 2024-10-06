#numpy
#provides the foundation data structures and operations for sCipy.These are arrays (ndarrays) that are efficient to define and manipulate data

#define an array
import numpy as np
list1 = [1, 2, 3]
print(list1)
array1 = np.array(list1)
print(array1)
print(array1.shape)

#access data
matrix1 = [[1, 2, 3], [4, 5, 6]]
print(matrix1)
array2 = np.array(matrix1)
print(array2)
print(array2.shape)
print("First row: %s" % array2[0])
print("Last row: %s" %array2[-1])
print("Specific row and col: %s" % array2[1,2])
print("Whole column: %s" %array2[:,1])

#arithmetic
array3 = np.array([2, 2, 2])
array4 = np.array([3, 3, 3])
print("Addition: %s" %(array3+array4))
print("Multiplication: %s" %(array3 * array4))
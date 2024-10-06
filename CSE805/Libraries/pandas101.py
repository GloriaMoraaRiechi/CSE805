#pandas provides data structures and functionality to quickly manipulate and analyze data
#need to understand the Series and DataFrame structures
from operator import index

#series: one dimensional array of data where the rows are labeled using a time axis
#pandas series is nothing but a column in an excel sheet
import numpy as np
import pandas as pd
array1 = np.array([1, 2, 3])
rowNames = ['a', 'b', 'c']
series1 = pd.Series(array1, rowNames)
print(series1)

# print(series1[2])
# print(series1['c'])

data2 = [1, 2, 3, 4]
series2 = pd.Series(data2)
print("Data: ")
print(series2)

#creating a series from an array

data3 = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
series3 = pd.Series(data3)
print(series3)

# #creating a series from a list
# list = ['g', 'e', 'e', 'k', 's']
# series4 = pd.Series(list)
# print(list)

#accessing elements from series with position (slice operation is used)
data5 = np.array(['g','e','e','k','s','f', 'o','r','g','e','e','k','s'])
series5 = pd.Series(data5)
print(series5[:5])  #accessing first five elements
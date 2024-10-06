#matplotlib: can be sued for creating plots and charts
#Plot(): calling a plotting function with some data
#call many functions to create the properties of the plot i.e labels and colors
#show(): make the plot visible

#line plot
import matplotlib.pyplot as plt
import numpy as np
from fontTools.unicodedata import block
from matplotlib.pyplot import figure

plt.figure(1)
array1 = np.array([1, 2, 3, 4, 6])
plt.plot(array1)
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Line plot')
plt.show(block=False)

#scatter plot
plt.figure(2)
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 9, 10, 27, 14])
plt.scatter(x,y)
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Scatter plot')
plt.show()
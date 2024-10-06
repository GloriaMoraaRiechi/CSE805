#MEAN
import numpy as np
from fontTools.unicodedata import block
from scipy.stats import alpha

speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]
x = np.mean(speed)
print(x)

#MEDIAN
median = np.median(speed)
print(median)

#MODE
from scipy import stats
modeSpeed = stats.mode(speed)
print(modeSpeed)

#STANDARD DEVIATION (describes how spread out the values are, square root of the variance)
#low std means that most of the numbers are close to the mean
#high std means that the values are spread out over a wide range
std = np.std(speed)
print("The standard deviation is: ", std)

#VARIANCE (measure of how spread out the values are)
var = np.var(speed)
print("The variance is: ", var)

#PERCENTILES (describes the value that a given percent of the values are lower than)
ages = [5,31,43,48,50,41,7,11,15,39,80,82,32,2,8,6,25,36,27,61,31]
per = np.percentile(ages, 90)
print(per)


#DATA DISTRIBUTION
num = np.random.uniform(0.0, 5.0, 250)
print(num)


import matplotlib.pyplot as plt

plt.figure(1)
plt.hist(num,5, color='blue', alpha=0.7)
plt.show(block=False)

plt.figure(2)
numbers = np.random.uniform(0.0, 5.0, 100000)
plt.hist(numbers, 100, color='green', alpha=0.7)
plt.show(block=False)

#normal data distribution
plt.figure(3)
ndd =  np.random.normal(5, 1, 100000)   #mean is 5 and the standard deviation is 1
plt.hist(ndd, 100, color='orange', alpha=0.7)
plt.show(block=False)
plt.title('Normal Data Distribution')


#SCATTER PLOT
#diagram where each value of the data set is represented by a dot
a = [5,7,8,7,2,17,2,9,4,11,12,9,6]
b = [99,86,87,88,111,86,103,87,94,78,77,85,86]
plt.figure(4)
plt.scatter(a,b)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot')
plt.show(block=False)


#RANDOM DATA DISTRIBUTIONS
c = np.random.normal(5.0, 1.0, 1000)
d = np.random.normal(10.0, 2.0, 1000)

plt.figure(5)
plt.scatter(c, d)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot, Random Data Distributions')
plt.show()

input("Press enter to close all the figures...")




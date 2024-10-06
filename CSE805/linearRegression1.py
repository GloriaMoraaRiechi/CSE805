#Regression
#used when you try to find the relationship between variables used to predict the outcome of future events

#LINEAR REGRESSION
#uses the relationship between data points to draw a straight line through all of them that is used to predict future events

import  matplotlib.pyplot as plt
from scipy import stats

x1 = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y1 = [99,86,87,88,111,86,103,87,94,78,77,85,86]

slope, intercept, r, p, std_err = stats.linregress(x1, y1)
print("Slope:", slope)   #how steep the line is(how much y changes per unit change in x
print("Intercept:", intercept)  #The point where the line crosses the y-axis, when x=0
print("P-value:", p)     #Statistical significance of the slope(whether the relationship is meaningful)
print("Standard Error of the slope:", std_err)   #How accurate the slope estimate is

def linreg(x1):
    return slope * x1 + intercept

myModel = list(map(linreg, x1))

plt.figure(1)
plt.scatter(x1, y1)
plt.plot(x1, myModel)  #Draw the line of linear regression
plt.title('Plot of Age vs Speed')
plt.xlabel('Age of the car')
plt.ylabel('Speed of the car')
plt.pause(60)
plt.show(block=False)

#COEFFICIENT OF RELATION, r
#ranges from -1 to 1 where 0 means no relationship, -1 and 1 means 100% relateed
print("R-value (correlation coefficient):", r)  #the strength of the linear relationship between x and y

#PREDICTING FUTURE VALUES
speed = linreg(10)
print(speed)







import numpy as np

print("Enter any list of numbers: ")

#take input from the user
listOfNumbers = input()

numberList = np.array(listOfNumbers.split(), dtype=float)

totalSum = 0
count = 0

for item in numberList:
    totalSum = totalSum + item
    count = count + 1

mean = totalSum/count
print(mean)
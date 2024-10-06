
#Mean Squared Error: Given by the calculating the average of the squared differences
#import sklearn
#from sklearn.matrices import mean_squared_error


#Step1: find the equation for the regression line

#y = 0.7*x - 0.1
#Step2: insert x values in the equation to get the predicted y values
#Step3: subtract the new predicted y values form the actual values to find the errors
#Step4: square the errors
#Step5: Sum upp all the errors
#Step6: Divide the sum by the totol number of values to find thhe average


# True positives (TP)
TP = 100

# True negatives (TN)
TN = 200

# False positives (FP)
FP = 50

# False negatives (FN)
FN = 150

# Confusion matrix
confusion_matrix = [[TP, FP], [FN, TN]]
print(confusion_matrix)
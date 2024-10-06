#NAIVE BAYES
#Assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import matthews_corrcoef
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve

#importing the dataset
dataset = pd.read_csv(r"C:\Users\glori\Desktop\CSE805\machinelearningMasteryWithPython\Social_Network_Ads.csv")
print(dataset.head())
#.iloc[] selects rows and columns by their index in a pandas dataframe by their index(integer location)
#.values converts the selected portion of the dataframe into a numpy array
X = dataset.iloc[:, 1: 4].values    #select all rows and columns 1 to 3(inclusive of 1 but not 4 in python)
y = dataset.iloc[:, -1].values      #select all the rows and just the last column
#print(dataset.shape[0])  #shows the number of rows and columns in the dataset
#print(dataset.shape[1])
print(X)

#TRANSFORMATION OF THE LABELS
#LabelEncoder is used to convert categorical data into numerical labels i.e male, female, yes, no, to 0 and 1
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()  #instance of the class le is created which will be used to transform categorical values in the dataset
#transformed values(now numerical) are assigned back to the first column of X
X[:, 0] = le.fit_transform(X[:, 0])   #first column is selected. .fit finds all unique categories in the first column, .transform converts each unique category into a corresponding numerical value
#print(X)

#SPLITTING THE DATASET INTO THE TRAINING SET AND TEST SET
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


#FEATURE SCALING
#Import standardScaler class which is used to standardize features by removing the mean and scaling unit variance
#standardization ensures that each feature contributes equally to the model
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()  #the scaler learns the mean and the standard deviation from the training daya and then uses those values to scale both the training and testing data
X_train = sc.fit_transform(X_train)  #fit computes the mean and standard deviation of each feature in the x_train and then
# scales the features using the calculated mean and standard deviation
#The resulting data will have a mean of 0 and a standard deviation of 1 for each feature
X_test = sc.transform(X_test)  #applies the same scaling bases on the training data to the test data ensuring consistent scaling across both datasets.

#TRAIN THE NAIVE BAYES MODEL ON THE TRAINING DATA
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()   #Create an instance of the GaussianNB class to have access to the attributes and methods of the class i.e fit(), predict()
classifier.fit(X_train, y_train)

#CROSS-VALIDATION USING kFOLD
kfold = KFold(n_splits=10, random_state=2, shuffle=True)
predicted = cross_val_predict(classifier, X_train, y_train, cv=kfold)
result = cross_val_score(classifier, X_train, y_train, cv=kfold, scoring='roc_auc')
confusion = confusion_matrix(y_train, predicted)

#MAKING THE CONFUSION MATRIX
from sklearn.metrics import confusion_matrix, accuracy_score
ac = accuracy_score(y_train, predicted)
cm = confusion_matrix(y_train, predicted) #computes the confusion matrix based on the true labels(y_train) and the predicted labels(predicted)
print(ac)

TP = cm[0,0]   #number of cases where the actual value was positive and the predicted value was also positive
TN = cm[1,1]   #number of cases where the actual value was negative and the predicted value was also negative
FP = cm[0,1]   #number of cases where the actual value was negative but the model incorrectly predicted as positive
FN = cm[1,0]   #number of cases where the actual value was positive but the model incorrectly predicted as negative


#SENSITIVITY/RECALL: The ability of the model to correctly identity positive cases
#measure of the ratio of actual positive cases that were correctly identified
#of all the actual positives, how many did the model identify correctly (true positive rate)
sensitivity = TP / float(FN + TP)
print("Sensitivity:" , sensitivity )

#SPECIFICITY/TRUE NEGATIVE RATE: The ability of the model to correctly identify negative cases
#of all the actual negatives, how many did the model correctly identify
specificity = TN / float(TN+FP)
print("Specificity: " , specificity)

#Area Under the curve: The ability of the model to distinguish between the positive and negative classes (higher auc indicates better model performance)
print("AUC: %3f" % (result.mean()))  #prints the mean AUC across all cross-validation folds

#Classification report: It includes;
#Precision: The proportion of true positives to all predicted positives
#Recall/Sensitivity: The proportion of true positives to all actual positives
#F1-Score: The harmonic mean of Precision and Recall (2pr/(p+r))
#Support: The number of occurrences of each class in the dataset(how many instances/data pointes of each class were present in the dataset that were used to evaluate the model)
print(cm)
print('Classification Report')
print(classification_report(y_train, predicted))

#ACCURACY: the proportion of correct predictions (both true positives and true negatives) to the total number of predictions
print("The accuracy is: " +str(ac))

#Matthews Correlation Coefficient (MCC): takes into account true and false positives and negatives
#useful for evaluating binary classification problems especially when the dataset is imbalanced
print('MCC is: '+str(matthews_corrcoef(y_train, predicted)))

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
ac = accuracy_score(y_test,y_pred)
com = confusion_matrix(y_test, y_pred)
print(ac)

TP1 = com[0, 0]
TN1 = com[1, 1]
FP1 = com[0, 1]
FN1 = com[1, 0]

sensitivity = TP1 / float(FN1 + TP1)
print("sensitivity:")
print(sensitivity)
specificity = TN1 / float(TN1 + FP1)
print("specificity ", specificity)




# Assuming the data and model are already trained
# - X_train, X_test, y_train, y_test, classifier, predicted, etc.

# Plot Confusion Matrix - Training Data
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=['Predicted: 0', 'Predicted: 1'], yticklabels=['Actual: 0', 'Actual: 1'])
plt.title('Confusion Matrix - Training Data')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)

# Plot Confusion Matrix - Test Data
plt.figure(figsize=(8,6))
sns.heatmap(com, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=['Predicted: 0', 'Predicted: 1'], yticklabels=['Actual: 0', 'Actual: 1'])
plt.title('Confusion Matrix - Test Data')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# ROC Curve - Test Data
y_prob = classifier.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='blue', label='ROC curve (AUC = %0.2f)' % roc_auc_score(y_test, y_prob))
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line (random model)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Precision-Recall Curve - Test Data
precision, recall, _ = precision_recall_curve(y_test, y_prob)

plt.figure(figsize=(8,6))
plt.plot(recall, precision, color='blue', label='Precision-Recall curve')
plt.xlabel('Recall (Sensitivity)')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best')
plt.show()

# Training metrics
train_accuracy = accuracy_score(y_train, predicted)
train_sensitivity = TP / float(FN + TP)
train_specificity = TN / float(TN + FP)

# Test metrics
test_accuracy = accuracy_score(y_test, y_pred)
test_sensitivity = TP1 / float(FN1 + TP1)
test_specificity = TN1 / float(TN1 + FP1)

# Plotting Accuracy, Sensitivity, Specificity
metrics = ['Accuracy', 'Sensitivity', 'Specificity']
train_values = [train_accuracy, train_sensitivity, train_specificity]
test_values = [test_accuracy, test_sensitivity, test_specificity]

metrics_df = pd.DataFrame({
    'Metrics': metrics,
    'Train': train_values,
    'Test': test_values
})

metrics_df.plot(x='Metrics', kind='bar', figsize=(8,6), title='Training vs Test Performance')
plt.ylabel('Score')
plt.show()

import time

plt.show(block=False)
time.sleep(1)  # Wait for 1 second

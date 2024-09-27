#K-nearest neighbor algorithm
#1. Create feature and target variable
#2. Split data into training and test data
#3. Generate a k-NN model using neighbors value
#4. Train or fit the data into the model
#5. Predict
from fontTools.unicodedata import block
#import the necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt


irisData = load_iris()

#Feature and target arrays
X = irisData.data
y = irisData.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

neighbors = np.arange(1, 9)  #creates an array of integers starting from 1 up to 9 to represent the number of neighbors k might be
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

#loop over K values
for i, k in enumerate(neighbors):
    knnModel1 = KNeighborsClassifier(n_neighbors=k)
    knnModel1.fit(X_train, y_train)

    #compute training and test data accuracy
    train_accuracy[i] = knnModel1.score(X_train, y_train)
    test_accuracy[i] = knnModel1.score(X_test, y_test)

#generate plot
plt.plot(neighbors, test_accuracy, label="Testing dataset accuracy")
plt.plot(neighbors, train_accuracy, label="Training dataset accuracy")
plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()



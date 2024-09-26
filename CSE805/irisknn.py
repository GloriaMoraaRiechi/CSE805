
import numpy as np   #used for handling arrays and numerical computations
import matplotlib.pyplot as plt   #used for plotting and handling numerical arrays
import pandas as pd     #used for data manipulation and analysis


#KNN CLASSIFICATION
#url to fetch the iris datacet
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
#defines the column names for the dataset. The first four are feature measurements of irir flowers and the fifth is species of the iris flower
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
#read the dataset from the specified url using pandas read_csv function, assigning the column names provided above
dataset = pd.read_csv(url, names=names)
print (dataset.head())  #Prints the first five rows of the dataset

#PRE-PROCESSING. iloc is used for extracting the necessary columns for training and testing the models
#input variables. Extract the input features(all rows and all columns except the last one)
X = dataset.iloc[:, :-1].values
#output variables. Extract the output labels(the species/class of the iris flower which is the last column) into why
y = dataset.iloc[:, 4].values   #selects the fifth column

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

#feature selection: creating scaler, an object that learns the mean and std of the data with fit() and transforms the data to standardize it with transform
from sklearn.preprocessing import StandardScaler #Standard scaler standardizes features by removing the mean and scaling them to unit variance
scaler = StandardScaler()                        #creates an instance/object(scaler) of StandardScaler which is a class that can be used to fit the data and transform the data
scaler.fit(X_train)                              #calculates the mean and the standard deviation of the features of X_train
#transform standardizes the features by subtracting the mean and dividing by the standard deviation
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Characterize
from sklearn.neighbors import KNeighborsClassifier   #used for implementing the K-Nearest neighbors algorithm
classifier = KNeighborsClassifier(n_neighbors=5)     #creates an instance KNeighborsClassifier and sets the number of neighbors. The algorithm will look at the 5 nearest data points to classify each test point
classifier.fit(X_train, y_train)                     #trains the KNN classifier on the training data and the corresponding labels which learns the patterns so that it can predict labels for unseen data
#the trained classifier is used to predict the labels for the test data
y_pred = classifier.predict(X_test)                  #generates a list of predicted class labels based on the model trained on the training data

#Evaluation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))              #shows how often the classifier predicted each class correctly or incorrectly
print(classification_report(y_test, y_pred))         #provides precision, recall and F1-score for each class
print(accuracy_score(y_test, y_pred)*100)            #Calculate the overall accuracy of the model(percentage of correct predictions)

#FINDING ERROR WITH CHANGED k
error = []
#calculating error fot k values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)       #Create a model with i neighbors
    knn.fit(X_train,y_train)                        #fit the model with training data
    pred_i = knn.predict(X_test)                    #predict the test set labels
    error.append(np.mean(pred_i != y_test))         #calculate the mean error(percentage of incorrect predictions)

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title('Plot of Error Rate vs K Value')
plt.xlabel('K Value')
plt.ylabel('mean Error')
plt.show()

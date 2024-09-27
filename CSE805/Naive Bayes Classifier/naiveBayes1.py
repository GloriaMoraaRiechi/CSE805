#load the iris dataset
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()

# #store the feature matrix(X) and response vector(y)
# X = iris.data
# y = iris.data

# Create a DataFrame for easier handling
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Features and target
X = df.drop('target', axis=1)  # Features
y = df['target']  # Target (Labels)

print(df)

#split X and y into training and test datasets
from sklearn.model_selection import  train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#train the model on the training set
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

#making predictions on the testing set
y_pred = gnb.predict(X_test)

#comparing actual response values (y_test) with the predicted values (y_pred)
from sklearn import metrics
print("Gaussian Naive Bayes model accuracy (in %) is", metrics.accuracy_score(y_test, y_pred)*100)









from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Create an SVM classifier with a linear kernel
svm = SVC(kernel='linear')

# Train the svm classifier on the training set
svm.fit(X_train, y_train)

# Make predictions
y_pred = svm.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)

#PARAMETERS TUNING
#GridSearchCV: function that takes the estimator(the model you want to tune), the parameter grid and other options like the cross validation strategy
from sklearn.model_selection import GridSearchCV

# define the parameter grid
paramGrid = {          #dictionary where keys are name of hyperparameters to tune and the values are lists of possible values to try
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'degree': [2, 3, 4],
    'coef0': [0.0, 0.1, 0.5],
    'gamma': ['scale', 'auto']
}

#create an svm classifier
svm = SVC()

# Perform grid search to find the best set of parameters
grid_search = GridSearchCV(svm, paramGrid, cv=5)
grid_search.fit(X_train, y_train) #fits the grid search object to the data (X and y) trying all possible combinations of hyperparameters defined in the paramGrid
best_parameters = grid_search.best_params_  #This attribute stores the best combination of hyperparameters found during the search
best_accuracy = grid_search.best_score_
# Print the best set of parameters and their accuracy
print('Best parameters: ', best_parameters)
print("best accuracy: ", best_accuracy)
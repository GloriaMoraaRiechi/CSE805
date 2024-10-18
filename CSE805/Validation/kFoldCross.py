from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Sample dataset
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([0, 0, 1, 1, 1])

# Initialize k-fold cross validation with 5 folds
kf = KFold(n_splits=5)

# Initialize model
model = LogisticRegression()

accuracies = []

# Perform K-FOLD Cross-Validation
for trainIndex, testIndex in kf.split(X):
    X_train, X_test = X[trainIndex], X[testIndex]
    y_train, y_test = y[trainIndex], y[testIndex]

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test fold
    y_pred = model.predict(X_test)

    # Evaluate Accuracy
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

# Print average accuracy across the folds
print(f"Average Accuracy: {np.mean(accuracies)}")


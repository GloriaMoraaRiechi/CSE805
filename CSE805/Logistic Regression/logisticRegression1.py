from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the iris dataset
iris = load_iris()
X = iris.data  # Features: Sepal length, Sepal width, Petal length, Petal width
y = iris.target  # Target: 3 classes (0, 1, 2)

# Convert this into a binary classification problem:
# We will classify whether the flower is Iris Setosa (class 0) or not (classes 1 and 2)
y_binary = (y == 0).astype(int)  # Setosa = 1, Not Setosa = 0

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# Initialize the logistic regression model
log_reg = LogisticRegression()

# Fit the model on the training data
log_reg.fit(X_train, y_train)

# Make predictions on the test data
y_pred = log_reg.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Output results
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix: \n{cm}")

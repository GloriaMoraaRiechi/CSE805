# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Example dataset (you can replace this with your actual data)
# Create a simple dataset with X as features and y as the target variable
data = {'x1': [1, 2, 3],
        'x2': [2, 3, 4],
        'y': [3, 6, 9]}

df = pd.DataFrame(data)

# Separate the features (X) and target (y)
X = df[['x1', 'x2']]  # Features
y = df['y']           # Target

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Ridge regression model with a regularization strength (alpha = 0.5 in this case)
ridge_model = Ridge(alpha=0.5)

# Fit the Ridge regression model
ridge_model.fit(X_train, y_train)

# Predict using the test set
y_pred = ridge_model.predict(X_test)

# Calculate mean squared error to evaluate performance
mse = mean_squared_error(y_test, y_pred)

# Print results
print(f"Coefficients: {ridge_model.coef_}")
print(f"Intercept: {ridge_model.intercept_}")
print(f"Mean Squared Error: {mse}")

# Regularization reduces the magnitude of the coefficients:
print(f"Regularized Coefficients: {ridge_model.coef_}")

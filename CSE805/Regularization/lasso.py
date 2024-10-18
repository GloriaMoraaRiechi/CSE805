# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Example dataset (you can replace this with your actual data)
data = {'x1': [1, 2, 3],
        'x2': [2, 3, 4],
        'y': [3, 6, 9]}

df = pd.DataFrame(data)

# Separate the features (X) and target (y)
X = df[['x1', 'x2']]  # Features
y = df['y']           # Target

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 7: Apply Lasso Regularization
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)

# STEP 8: Make predictions and evaluate
y_pred = lasso_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared Score: {lasso_model.score(X_test, y_test):.4f}")

# STEP 9: Inspect coefficients
coefficients = pd.Series(lasso_model.coef_, index=X.columns)
print("Lasso coefficients:\n", coefficients)

# Plot the coefficients
plt.figure(figsize=(10, 6))
coefficients.plot(kind='bar')
plt.title("Lasso Coefficients")
plt.show()

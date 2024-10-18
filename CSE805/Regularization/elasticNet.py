# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Example dataset (you can replace this with your actual data)
data = {'x1': [1, 2, 3, 4, 5],
        'x2': [5, 4, 3, 2, 1],
        'y': [2, 3, 5, 7, 11]}

df = pd.DataFrame(data)

# Separate the features (X) and target (y)
X = df[['x1', 'x2']]  # Features
y = df['y']           # Target

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Elastic Net model with regularization strengths (alpha and l1_ratio)
elastic_net_model = ElasticNet(alpha=0.1, l1_ratio=0.5)  # l1_ratio = 0.5 is a mix of Lasso and Ridge

# Fit the Elastic Net model
elastic_net_model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = elastic_net_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared Score: {elastic_net_model.score(X_test, y_test):.4f}")

# Inspect coefficients
coefficients = pd.Series(elastic_net_model.coef_, index=X.columns)
print("Elastic Net coefficients:\n", coefficients)

# Plot the coefficients
plt.figure(figsize=(10, 6))
coefficients.plot(kind='bar')
plt.title("Elastic Net Coefficients")
plt.show()

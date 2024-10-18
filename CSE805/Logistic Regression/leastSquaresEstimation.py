import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate some example data
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # Independent variable
y = 2.5 * X + np.random.randn(100, 1) * 2  # Dependent variable with noise

# Fit the linear regression model using least squares
model = LinearRegression()
model.fit(X, y)

# Get the estimated parameters (intercept and slope)
intercept = model.intercept_
slope = model.coef_

# Predict the values
y_pred = model.predict(X)

# Evaluate the model (mean squared error)
mse = mean_squared_error(y, y_pred)

# Print the results
print(f"Intercept (beta_0): {intercept}")
print(f"Slope (beta_1): {slope}")
print(f"Mean Squared Error: {mse}")

# Plot the data and the fitted line
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, y_pred, color='red', label='Fitted line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

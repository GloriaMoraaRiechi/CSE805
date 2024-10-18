# 1. Simple Stacking
# Predictions from various base models are stacked together and used as input feature for a meta model which subsequently makes the final predictions
# Simple stacking can be particularly useful when you have models with varied predictive powers, enabling the meta-model to learn how to weigh the input from each model based on their reliability.
import numpy as np
# 2. Cross-Validation Stacking
# The training set is split into K folds and the base models are trained k times on k-1 fold predicting the left out fold each time
# These out-of-fold predictions for each model are stacked and used as features for the meta model
# Ensures that every observation has the chances of appearing in the training and the test set and less prone to overfitting
# Ideal in scenarios where model stability and generalization are crucial, since using out-of-fold predictions for training the meta-model helps in reducing overfitting.

# 1. DATA PREPARATION
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load the dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Split data into training, validation and testing sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

#2. BUILD BASE MODELS
# Goal is to create a diverse set of models that will make different types of predictions based on different aspects of data
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

# Initialize base models
rf = RandomForestRegressor(n_estimators=10, random_state=42)
knn = KNeighborsRegressor(n_neighbors=5)
lr = LinearRegression()

# Fit base models on the training data
rf.fit(X_train, y_train)
knn.fit(X_train, y_train)
lr.fit(X_train, y_train)

# Generate base model predictions on training and validation data
rf_pred_train = rf.predict(X_train)
knn_pred_train = knn.predict(X_train)
lr_pred_train = lr.predict(X_train)

rf_pred_val = rf.predict(X_val)
knn_pred_val = knn.predict(X_val)
lr_pred_val = lr.predict(X_val)

# 3. BUILD THE META MODEL
# Takes as input the predictions made by each of the base models on the validation data and outputs a final prediction for each observation
# Combine base model predictions into meta features
meta_features_train = np.column_stack((rf_pred_train, knn_pred_train, lr_pred_train))
meta_features_val = np.column_stack((rf_pred_val, knn_pred_val, lr_pred_val))

# Initialize meta model
ridge = Ridge(alpha=0.5)

# 4. COMBINE BASE NAD META MODELS
# Fit the model on meta features and training target variable
ridge.fit(meta_features_train, y_train)

# generate meta model predictions on validation set
meta_pred_val = ridge.predict(meta_features_val)

from sklearn.metrics import mean_squared_error

# calculate MSE on validation set
mse = mean_squared_error(y_val, meta_pred_val)
print("MSE on validation set: {:.4f}". format(mse))
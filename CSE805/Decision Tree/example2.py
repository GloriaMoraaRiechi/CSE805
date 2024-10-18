# Importing necessary libraries
import numpy as np               # For array manipulation
import matplotlib.pyplot as plt  # For plotting graphs
import pandas as pd              # For handling datasets (though not used in this example)
import graphviz

# Import dataset (using a numpy array here instead of reading a CSV file)
dataset = np.array(
    [['Asset Flip', 100, 1000],
     ['Text Based', 500, 3000],
     ['Visual Novel', 1500, 5000],
     ['2D Pixel Art', 3500, 8000],
     ['2D Vector Art', 5000, 6500],
     ['Strategy', 6000, 7000],
     ['First Person Shooter', 8000, 15000],
     ['Simulator', 9500, 20000],
     ['Racing', 12000, 21000],
     ['RPG', 14000, 25000],
     ['Sandbox', 15500, 27000],
     ['Open-World', 16500, 30000],
     ['MMOFPS', 25000, 52000],
     ['MMORPG', 30000, 80000]
    ])

# Select features (X) - column 1 (Production Cost)
X = dataset[:, 1:2].astype(int)  # Selecting the second column (index 1) as the feature
# print(X)  # Uncomment to see the features

# Select labels (y) - column 2 (Profit)
y = dataset[:, 2].astype(int)  # Selecting the third column (index 2) as the target/label
# print(y)  # Uncomment to see the labels

# Importing Decision Tree Regressor from scikit-learn
from sklearn.tree import DecisionTreeRegressor

# Create a regressor object
regressor = DecisionTreeRegressor(random_state=0)

# Fit the regressor to the data (X and y)
regressor.fit(X, y)

# Predicting a new value (for example, Production Cost = 3750)
y_pred = regressor.predict([[3750]])

# Print the predicted price
# Extract the single value from the array and format it as an integer
print("Predicted price: %d" % int(y_pred[0]))  # Fix: Use y_pred[0] to get the scalar

# Create a range of values for plotting the decision tree (for smooth curve)
X_grid = np.arange(min(X).item(), max(X).item(), 0.01).reshape(-1, 1)  # Ensure min and max return scalars

# Scatter plot for the original data (red dots)
plt.scatter(X, y, color='red')

# Plot the predicted data (blue line)
plt.plot(X_grid, regressor.predict(X_grid), color='blue')

# Set title and labels for the plot
plt.title('Profit vs Production Cost (Decision Tree Regression)')
plt.xlabel('Production Cost')
plt.ylabel('Profit')

# Show the plot
plt.show()

# import export_graphviz
from sklearn.tree import export_graphviz

# export the decision tree to a tree.dot file
# for visualizing the plot easily anywhere
export_graphviz(regressor, out_file='tree.dot',
                feature_names=['Production Cost'])

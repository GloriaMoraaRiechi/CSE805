from sklearn import svm
import numpy as np

# Data points (features)
X = np.array([[1, 2], [2, 3], [2, 1], [3, 2]])

# Class labels
y = np.array([1, 1, -1, -1])

# Create an SVM classifier with a linear kernel
clf = svm.SVC(kernel='linear')

# Train the model
clf.fit(X, y)

# Get the optimal weights (w) and bias (b)
w = clf.coef_
b = clf.intercept_

print("Weights (w):", w)
print("Bias (b):", b)

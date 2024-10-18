import pandas as pd  # used to load and manipulate data and for one-hot encoding
import numpy as np  # data manipulations
import matplotlib.pyplot as plt  # drawing graphs
from sklearn.utils import resample  # downsample the dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV  # will do cross-validation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.decomposition import PCA

# IMPORT THE DATA
df = pd.read_excel(r"C:\Users\glori\Desktop\CSE805\CSE805\Support Vector Machines\default of credit card clients.xls", header=1)

# CLEANING THE DATA
df.rename({'default payment next month': 'DEFAULT'}, axis='columns', inplace=True)  # modify column name
df.drop('ID', axis=1, inplace=True)  # set axis=0 to remove rows, axis=1 to remove columns

# Drop the rows with missing values
df_no_missing = df.loc[(df['EDUCATION'] != 0) & (df['MARRIAGE'] != 0)]

# DOWN-SAMPLE THE DATA
df_no_default = df_no_missing[df_no_missing['DEFAULT'] == 0]
df_default = df_no_missing[df_no_missing['DEFAULT'] == 1]

# Down-sample those who did not default
df_no_default_downSampled = resample(df_no_default, replace=False, n_samples=1000, random_state=42)
# Down-sample those who defaulted
df_default_downSampled = resample(df_default, replace=False, n_samples=1000, random_state=42)

# Merge the two dataframes
df_downSample = pd.concat([df_default_downSampled, df_no_default_downSampled])

# FORMAT THE DATA
# Independent variables, used to make predictions
X = df_downSample.drop('DEFAULT', axis=1).copy()
# Dependent variable, variable we want to predict
y = df_downSample['DEFAULT'].copy()

# One Hot Encoding
X_encoded = pd.get_dummies(X, columns=['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']).astype(int)

# Centering and Scaling the data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit the training data
X_test_scaled = scaler.transform(X_test)        # transform the test data

# APPLY PCA
pca = PCA(n_components=2)  # Set to reduce the dimensionality to 2
X_train_pca = pca.fit_transform(X_train_scaled)

# BUILD A PRELIMINARY SUPPORT VECTOR MACHINE
clf_svm = SVC(random_state=42)  # creates an untrained shell of a support vector classifier
clf_svm.fit(X_train_scaled, y_train)

# OPTIMIZE PARAMETERS WITH CROSS VALIDATION AND GridSearchCV()
param_grid = [
    {'C': [0.5, 1, 10, 100],
     'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001],
     'kernel': ['rbf']},
]

optimal_params = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', verbose=0)
optimal_params.fit(X_train_scaled, y_train)

# BUILD, EVALUATE, DRAW AND INTERPRET THE FINAL SUPPORT VECTOR MACHINE
optimal_params = optimal_params.best_params_
clf_svm = SVC(random_state=42, C=optimal_params['C'], gamma=optimal_params['gamma'])
clf_svm.fit(X_train_scaled, y_train)

# DECISION BOUNDARY
# Create a mesh grid to plot the decision boundary
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Predict class labels over the grid using the optimized SVM model
Z = clf_svm.predict(scaler.transform(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])))
Z = Z.reshape(xx.shape)  # Reshape the predictions to match the grid shape

# Plot the decision boundary
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)

# Plot the training points
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, edgecolors='k', cmap=plt.cm.RdYlBu, marker='o', alpha=0.8)

# Add a color bar
plt.colorbar(scatter)

# Add labels and title
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('SVM Decision Boundary with PCA-transformed Data')

# Show the plot
plt.show()

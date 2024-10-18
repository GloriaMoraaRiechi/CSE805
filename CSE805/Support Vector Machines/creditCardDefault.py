import pandas as pd #used to load and manipulate data and for one-hot encoding
import numpy as np  #data manipulations
import matplotlib.pyplot as plt #drawing graphs
from sklearn.utils import resample #downsample the dataset
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV #will do cross validation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# IMPORT THE DATA
df = pd.read_excel(r"C:\Users\glori\Desktop\CSE805\CSE805\Support Vector Machines\default of credit card clients.xls", header=1)
print(df.head())

# 1. CLEANING THE DATA
df.rename({'default payment next month' : 'DEFAULT'}, axis='columns', inplace=True)  #modify column name
df.drop('ID',axis=1, inplace=True)  # set axis=0 to remove rows, axis=1 to remove columns

# IDENTIFYING MISSING DATA
#print(df.dtypes)

print(df['SEX'].unique())
print(df['EDUCATION'].unique())
print(df['MARRIAGE'].unique())

print(len(df))
# Print number of rows that contain 0 in education and marriage
print(len(df.loc[(df['EDUCATION'] == 0) | (df['MARRIAGE'] == 0)]))

# Drop the 68 rows with missing values
df_no_missing = df.loc[(df['EDUCATION'] != 0) & (df['MARRIAGE'] != 0)]
print(df_no_missing['SEX'].unique())
print(df_no_missing['EDUCATION'].unique())
print(df_no_missing['MARRIAGE'].unique())
#print((df_no_missing.head()))

# 2. DATA REDUCTION
# DOWN-SAMPLE THE DATA(reduce the size of data by decreasing the data points / samples)
print(len(df_no_missing))  #prints 29932 samples

# Split the data into two dataframes
df_no_default = df_no_missing[df_no_missing['DEFAULT'] == 0]
print("Did not default:")
print(df_no_default.head())
df_default = df_no_missing[df_no_missing['DEFAULT'] == 1]
print("Defaulted:")
print(df_default.head())

# Down-sample those who did not default
df_no_default_downSampled = resample(df_no_default, replace=False, n_samples=1000, random_state=42)
print(len(df_no_default_downSampled))

# Down-sample those who defaulted
df_default_downSampled = resample(df_default, replace=False, n_samples=1000, random_state=42)
print(len(df_default_downSampled))

# Merge the two dataframes
df_downSample = pd.concat([df_default_downSampled, df_no_default_downSampled])
print(len(df_downSample))

# 3. DATA TRANSFORMATION AND DATA DISCRETIZATION
# FORMAT THE DATA
# 1) Split the data into Dependent and Independent variables

# Independent variables, used to make predictions
X = df_downSample.drop('DEFAULT', axis=1).copy()
print('The independent variables:')
print(X.head())

# Dependent variable, variable we want to predict
y = df_downSample['DEFAULT'].copy()
print('The dependent variable:')
print(y.head())

# 2) One Hot Encoding (Converts categorical data into a numerical format that algorithms can process, binary vectors)
X_encoded = pd.get_dummies(X, columns=['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6' ]).astype(int)
print(X_encoded.head())

# 3) Centering and Scaling the data, each column has a mean of 0 and a standard deviation of 1
# Split the data into training and test sets to avoid data leakage which occurs when information about the training set corrupts or influences the testing dataset
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  #fit the training data
X_test_scaled = scaler.transform(X_test)        #transform the test data

# BUILD A PRELIMINARY SUPPORT VECTOR MACHINE
clf_svm = SVC(random_state=42)  #creates an untrained shell of a support vector classifier
clf_svm.fit(X_train_scaled, y_train)

print("\nDefault Settings:")
print(clf_svm.get_params())

#Make predictions on the scaled test set
y_pred = clf_svm.predict(X_test_scaled)

# compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Extract TP, TN, FP, FN from the confusion matrix
TN, FP, FN, TP = cm.ravel()  #Flattens the confusion matrix into a one-dimensional array, TN at index 0, FP at 1, FN at 2 and TP at 3

# Display the results
print(f"True Positives (TP): {TP}")
print(f"True Negatives (TN): {TN}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")


# Create a confusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Did not default", "defaulted"])

# Plot the confusion matrix
disp.plot(cmap=plt.cm.Blues, values_format='d')  # 'd' for integer format
plt.title('Confusion Matrix')
plt.show()

# OPTIMIZE PARAMETERS WITH CROSS VALIDATION AND GridSearchCV()
#Finding the best gamma and potentially the regularization parameter, C to improve the accuracy with the testing dataset
#C: regularization parameter that contols the trade-off between maximizing and minimizing classification errors. determines how much you want the model to avoid mis-classifying training data points
param_grid = [
    {'C': [0.5, 1, 10, 100],
     'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001],
     'kernel': ['rbf']},
]

optimal_params = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', verbose=0)

optimal_params.fit(X_train_scaled, y_train)
print(optimal_params.best_params_)

# BUILD, EVALUATE, DRAW AND INTERPRET THE FINAL SUPPORT VECTOR MACHINE
optimal_params = optimal_params.best_params_
clf_svm = SVC(random_state=42, C=optimal_params['C'], gamma=optimal_params['gamma'])
clf_svm.fit(X_train_scaled, y_train)

print("\nFinal Model Settings:")
print(clf_svm.get_params())

# Make predictions on the scaled test set using the optimized model
y_pred_optimized = clf_svm.predict(X_test_scaled)

# compute the confusion matrix
cm = confusion_matrix(y_test, y_pred_optimized)

# Extract TP, TN, FP, FN from the confusion matrix
TN, FP, FN, TP = cm.ravel()  #Flattens the confusion matrix into a one-dimensional array, TN at index 0, FP at 1, FN at 2 and TP at 3

# Display the results
print(f"True Positives (TP): {TP}")
print(f"True Negatives (TN): {TN}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")


# Create a confusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Did not default", "defaulted"])

# Plot the confusion matrix
disp.plot(cmap=plt.cm.Blues, values_format='d')  # 'd' for integer format
plt.title('Confusion Matrix - Optimized Model')
plt.show()


# DECISION BOUNDARY
print(len(df_downSample.columns))

pca = PCA()
X_train_pca = pca.fit_transform(X_train_scaled)

per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
labels = [str(x) for x in range(1, len(per_var)+1)]

plt.bar(x=range(1, len(per_var)+1), height=per_var)
plt.tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=False,
    labelbottom=False
)

plt.ylabel('Percentage of explained variance')
plt.xlabel('Principal Components')
plt.title('Scree Plot')
plt.show()

train_pc1_coords = X_train_pca[:, 0]
train_pc2_coords = X_train_pca[:, 1]

pca = PCA(n_components=2)  # or whatever number of components you want
pca_train_scaled = pca.fit_transform(X_train_scaled)

param_grid = [
    {
        'C': [1, 100, 1000],
        'gamma': ['scale', 1, 0.1, 0.01, 0.001],
        'kernel': ['rbf']
    }
]

optimal_params = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', verbose=0)

optimal_params.fit(pca_train_scaled, y_train)
print(optimal_params.best_params_)

# Create a mesh grid to plot the decision boundary
x_min, x_max = pca_train_scaled[:, 0].min() - 1, pca_train_scaled[:, 0].max() + 1
y_min, y_max = pca_train_scaled[:, 1].min() - 1, pca_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Predict class labels over the grid using the optimized SVM model
Z = clf_svm.predict(np.c_[xx.ravel(), yy.ravel()])  # Combine the grid into a feature array
Z = Z.reshape(xx.shape)  # Reshape the predictions to match the grid shape

# Plot the decision boundary
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)

# Plot the training points
scatter = plt.scatter(pca_train_scaled[:, 0], pca_train_scaled[:, 1], c=y_train, edgecolors='k', cmap=plt.cm.RdYlBu, marker='o')

# Add a color bar
plt.colorbar(scatter)

# Add labels and title
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('SVM Decision Boundary with PCA-transformed Data')

# Show the plot
plt.show()


from unittest.mock import inplace

import pandas as pd  # used to load and manipulate data and for one-hot encoding
import numpy as np  # data manipulations
import matplotlib.pyplot as plt  # drawing graphs
from sklearn.utils import resample  # downsample the dataset
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV  # will do cross-validation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import seaborn as sns
from statsmodels.tools import categorical

# Step1. Import the data
df = pd.read_csv(r"C:\Users\glori\Desktop\CSE805\CSE805\Support Vector Machines\Air_Quality.csv")
print(df.head())

# a. Rename columns with two words to camelCase
def to_camel_case(s):
    """Convert a string to camelCase."""
    components = s.split()  # Split the string by spaces
    return components[0].lower() + ''.join(x.capitalize() for x in components[1:])  # Apply camelCase

# Apply the function to all column names
df.columns = [to_camel_case(col) for col in df.columns]

# b. Drop irrelevant columns
df.drop('uniqueId', axis=1, inplace=True)
df.drop('indicatorId', axis=1, inplace=True)
print(df.head())
print(df.columns)

print(df.info())
print(df.describe())
print(df.isnull().sum())
print("Columns")
print(df.columns)
print(len(df.columns))  # To check the number of columns

# c. Handle missing values
# First, replace '\t?' with NaN
df = df.replace('\t?', np.nan)
print(df.dtypes)

# Convert numeric columns to float
# Separate numeric and categorical columns
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()  # Numeric columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()  # Categorical columns

# Display the separated columns
print("Numeric Columns:")
print(numeric_cols)

print("\nCategorical Columns:")
print(categorical_cols)

# Convert numeric columns to float
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# # Impute missing values
# numeric_imputer = SimpleImputer(strategy='mean')
# categorical_imputer = SimpleImputer(strategy='most_frequent')
#
# df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
# df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])
print(df.head())

# 4. Check for outliers
def plot_boxplot(dataframe, features):
    fig, axes = plt.subplots(len(features) // 3 + 1, 3, figsize=(20, 5 * (len(features) // 3 + 1)))
    for i, feature in enumerate(features):
        sns.boxplot(x=dataframe[feature], ax=axes[i // 3, i % 3])
    plt.tight_layout()
    plt.show()

plot_boxplot(df, numeric_cols)

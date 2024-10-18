import pandas as pd
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import seaborn as sns


# STEP1. IMPORT THE DATA
df = pd.read_csv(r"C:\Users\glori\Desktop\CSE805\CSE805\Support Vector Machines\CDK_DataCopy.csv")

original_counts = df['Diagnosis'].value_counts()
print(original_counts)


# STEP2. CLEAN THE DATA
# Drop irrelevant columns
df.drop('PatientID', axis=1, inplace=True)
columns = [col for col in df.columns if col not in ['Diagnosis', 'DoctorInCharge']]
columns.append('Diagnosis')
df = df[columns]

# Explore the data
print(df.info())
print(df.describe())       #Give the description of the data, mean, std, min, max, quartiles of each feature
print(df.isnull().sum())   #calculate the sum of missing values in each column
print("Columns")
print(df.columns)
print(len(df.columns))

# CHECK FOR OUTLIERS
# Calculate Z-scores for all columns
z_scores = df[columns].apply(zscore)

# Identify outliers where |z| > 3
outliers_z = (z_scores.abs() > 3).sum()
print(outliers_z[outliers_z > 0])


# Handle the outliers using IQR
# 1. HeavyMetalsExposure (73 outliers)
Q1 = df['HeavyMetalsExposure'].quantile(0.25)
Q3 = df['HeavyMetalsExposure'].quantile(0.75)
IQR = Q3 - Q1

# Define lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Capping the outliers
df['HeavyMetalsExposure'] = df['HeavyMetalsExposure'].clip(lower=lower_bound, upper=upper_bound)

# 2. Diagnosis(135 outliers)
# Calculate the IQR for Diagnosis
Q1 = df['Diagnosis'].quantile(0.25)
Q3 = df['Diagnosis'].quantile(0.75)
IQR = Q3 - Q1

# Define lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove rows where Diagnosis is an outlier
df = df[(df['Diagnosis'] >= lower_bound) & (df['Diagnosis'] <= upper_bound)]

# Confirm if there are any outliers left
# Apply Z-scores to columns after handling outliers
z_scores = df[columns].apply(zscore)

# Identify outliers where |z| > 3
outliers_z = (z_scores.abs() > 3).sum()

# Print the features with outlier counts using Z-scores
for feature, count in zip(columns, outliers_z):
    print(f"{feature}: {count} outliers (Z-score method)")

# Save the cleaned dataset
df.to_csv(r'C:\Users\glori\Desktop\CSE805\CSE805\Graduate Seminar\cleanedCDK.csv', index=False)

print("Data cleaning completed. Cleaned data saved to 'cleanedCDK.csv'")


# STEP3. NORMALIZE THE DATA
# Scale each feature so that it has a mean of 0 and a standard deviation of 1 to make features more comparable

from scipy.stats import zscore

# Identify numeric columns, excluding 'Diagnosis'
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
numeric_columns = numeric_columns.drop('Diagnosis', errors='ignore')

# Apply Z-score normalization to the remaining numeric columns
df[numeric_columns] = df[numeric_columns].apply(zscore)

# Print to verify
print(df.describe())


# STEP4. FEATURE SELECTION
from sklearn.feature_selection import VarianceThreshold

# Adjusting threshold to remove low variance features
selector = VarianceThreshold(threshold=0.01)  # Lower threshold for low-variance feature removal
X_selected = selector.fit_transform(df.drop('Diagnosis', axis=1))
print("Selected features after variance thresholding:", X_selected.shape)

# Get the indices of features selected by the variance threshold
selected_features = selector.get_support(indices=True)
selected_columns = df.drop('Diagnosis', axis=1).columns[selected_features]

# Display the selected columns
print("Selected Features:\n", selected_columns)
print(len(selected_columns))


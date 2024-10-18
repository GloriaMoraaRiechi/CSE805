import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.ma.extras import column_stack
from sklearn.impute import SimpleImputer
import seaborn as sns

# Import the Data
df = pd.read_csv(r"C:\Users\glori\Desktop\CSE805\CSE805\Support Vector Machines\CDK_Data.csv")
print(df.head())
print(f"Initial number of columns: {len(df.columns)}")
print(df.columns)
# STEP 1: CLEAN THE DATA

# Drop irrelevant columns
if 'patientID' in df.columns:
    df.drop('PatientID', axis=1, inplace=True)

# Handle missing values
# First, replace '\t?' with NaN
df.replace('\t?', np.nan, inplace=True)

# Convert numeric columns to float
numeric_columns = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar',
                   'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
                   'potassium', 'hemoglobin', 'packed_cell_volume', 'white_blood_cell_count',
                   'red_blood_cell_count']

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Impute missing values for numeric columns
numeric_imputer = SimpleImputer(strategy='mean')
df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])

# Impute missing values for categorical columns
categorical_columns = ['red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria',
                       'hypertension', 'diabetes_mellitus', 'coronary_artery_disease',
                       'appetite', 'pedal_edema', 'anemia']

categorical_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])

# Handle inconsistent categorical values
df['class'] = df['class'].replace({'ckd\t': 'ckd', 'notckd': 'notckd'})
df['class'] = df['class'].map({'ckd': 1, 'notckd': 0})

# Clean categorical columns
for col in categorical_columns:
    df[col] = df[col].str.lower().str.strip()

# Map binary categorical variables to 0 and 1
binary_map = {'yes': 1, 'no': 0, 'present': 1, 'notpresent': 0, 'abnormal': 1, 'normal': 0,
              'poor': 1, 'good': 0, '\tyes': 1, '\tno': 0}

for col in categorical_columns:
    df[col] = df[col].map(binary_map)

# Check for outliers using boxplots
def plot_boxplot(dataframe, features):
    fig, axes = plt.subplots(len(features) // 3 + 1, 3, figsize=(20, 5 * (len(features) // 3 + 1)))
    for i, feature in enumerate(features):
        sns.boxplot(x=dataframe[feature], ax=axes[i // 3, i % 3])
    plt.tight_layout()
    plt.show()

plot_boxplot(df, numeric_columns)

# Handle outliers for 'hemoglobin'
Q1 = df['hemoglobin'].quantile(0.25)
Q3 = df['hemoglobin'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - (1.5 * IQR)
upper_bound = Q3 + (1.5 * IQR)
df['hemoglobin'] = df['hemoglobin'].clip(lower_bound, upper_bound)

# Final check for missing values and data types
print("Missing values after cleaning:")
print(df.isnull().sum())

print("Data types:")
print(df.dtypes)

# Save cleaned dataset
df.to_csv(r'"C:\Users\glori\Desktop\CSE805\CSE805\Support Vector Machines"\cleaned_kidney_disease.csv', index=False)
df.to_csv('cleaned_kidney_disease.csv', index=False)
print("Data cleaning completed. Cleaned data saved to 'cleaned_kidney_disease.csv'")

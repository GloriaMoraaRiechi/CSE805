import pandas as pd  # used to load and manipulate data and for one-hot encoding
import numpy as np  # data manipulations
import matplotlib.pyplot as plt  # drawing graphs
from sklearn.impute import SimpleImputer
import seaborn as sns

# Step 1: Import the data
df = pd.read_csv(r"C:\Users\glori\Desktop\CSE805\CSE805\class project\Air_Quality.csv")
# print(df.head())

# a. Rename columns with two words to camelCase
def to_camel_case(s):
    """Convert a string to camelCase."""
    components = s.split()  # Split the string by spaces
    return components[0].lower() + ''.join(x.capitalize() for x in components[1:])  # Apply camelCase

# Apply the function to all column names
# print(df.columns)
df.columns = [to_camel_case(col) for col in df.columns]
# print(df.columns)

# b. Drop irrelevant columns
columns_to_drop = ['uniqueId', 'indicatorId']
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)
# print(df.columns)


# c. Check DataFrame info, description, null values
#print(df.info())
#print(df.describe())
print(df.isnull().sum())

# d. Handle missing values
# Check for missing values
print(df.isnull().sum())

# missingValues = df.isna()
# missingCounts = df.isna().sum()
# # print("Missing Values: ", missingValues)
# # print("Missing counts: ", missingCounts)

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

# Filter numeric columns to only those with at least one non-null value
numeric_cols = [col for col in numeric_cols if df[col].notnull().any()]

# Impute missing values
numeric_imputer = SimpleImputer(strategy='mean')
categorical_imputer = SimpleImputer(strategy='most_frequent')

# Apply imputation only if there are valid columns
if numeric_cols:  # Check if there are numeric columns left for imputation
    df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])

if categorical_cols:  # Check if there are categorical columns left for imputation
    df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])

print(df.head())

# 4. Check for outliers
def plot_boxplot(dataframe, features):
    fig, axes = plt.subplots(len(features) // 3 + 1, 3, figsize=(20, 5 * (len(features) // 3 + 1)))
    for i, feature in enumerate(features):
        ax = axes[i // 3, i % 3] if len(features) > 3 else axes[i]  # Adjust for fewer features
        sns.boxplot(x=dataframe[feature], ax=ax)
        ax.set_title(feature)
    plt.tight_layout()
    plt.show()

# Plot boxplots for numeric columns to check for outliers
plot_boxplot(df, numeric_cols)

# 5. Handle outliers using IQR method
def clip_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower_bound, upper_bound)

# Clip outliers in numeric columns
clip_outliers(df, numeric_cols)

# Check the result after handling outliers
print(df.describe())

# 6. Final check
print(df.isnull().sum())
print(df.dtypes)

# 7. Save cleaned dataset
df.to_csv(r'C:\Users\glori\Desktop\CSE805\CSE805\Support Vector Machines\cleaned_air_quality.csv', index=False)

print("Data cleaning completed. Cleaned data saved to 'cleaned_air_quality'")



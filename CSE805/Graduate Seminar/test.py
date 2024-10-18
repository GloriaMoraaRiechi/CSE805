import pandas as pd
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# STEP 1: IMPORT THE DATA
df = pd.read_csv(r"C:\Users\glori\Desktop\CSE805\CSE805\Support Vector Machines\CDK_DataCopy.csv")

# Display original counts of Diagnosis
original_counts = df['Diagnosis'].value_counts()
print("Original counts:\n", original_counts)

# STEP 2: CLEAN THE DATA
# Drop irrelevant columns
df.drop('PatientID', axis=1, inplace=True)
columns = [col for col in df.columns if col not in ['Diagnosis', 'DoctorInCharge']]
columns.append('Diagnosis')
df = df[columns]

# Handle missing values
imputer = SimpleImputer(strategy='mean')  # Impute missing values with mean
df[columns[:-1]] = imputer.fit_transform(df[columns[:-1]])

# Explore the data
print(df.info())
print(df.describe())  # Describe the data: mean, std, min, max, quartiles
print(df.isnull().sum())  # Check for remaining missing values

# CHECK FOR OUTLIERS
# Calculate Z-scores for all columns
z_scores = df[columns].apply(zscore)

# Identify outliers where |z| > 3
outliers_z = (z_scores.abs() > 3).sum()
print("Outliers using Z-score:\n", outliers_z[outliers_z > 0])

# Handle outliers using IQR for HeavyMetalsExposure
Q1 = df['HeavyMetalsExposure'].quantile(0.25)
Q3 = df['HeavyMetalsExposure'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df['HeavyMetalsExposure'] = df['HeavyMetalsExposure'].clip(lower=lower_bound, upper=upper_bound)

# Confirm if there are any outliers left after capping
z_scores = df[columns].apply(zscore)
outliers_z = (z_scores.abs() > 3).sum()
for feature, count in zip(columns, outliers_z):
    print(f"{feature}: {count} outliers (Z-score method)")

# STEP 3: NORMALIZE THE DATA
# Identify numeric columns, excluding 'Diagnosis'
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
numeric_columns = numeric_columns.drop('Diagnosis', errors='ignore')

# Apply Z-score normalization
df[numeric_columns] = df[numeric_columns].apply(zscore)

# Print to verify normalization
print("Normalized Data Description:\n", df.describe())

# STEP 4: FEATURE SELECTION
from sklearn.feature_selection import VarianceThreshold

# Adjusting threshold to remove low variance features
selector = VarianceThreshold(threshold=0.01)
X_selected = selector.fit_transform(df.drop('Diagnosis', axis=1))
print("Selected features after variance thresholding:", X_selected.shape)

# Get the indices of features selected by the variance threshold
selected_features = selector.get_support(indices=True)
selected_columns = df.drop('Diagnosis', axis=1).columns[selected_features]

# Display the selected columns
print("Selected Features:\n", selected_columns)

# STEP 5: TRAINING THE SVM MODEL
# Prepare features and labels
X = df[selected_columns]
y = df['Diagnosis']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the SVM model
svm_model = SVC()
svm_model.fit(X_train, y_train)  # Train the model

# Make predictions
y_pred = svm_model.predict(X_test)

# STEP 6: EVALUATE THE MODEL
# Print classification report and confusion matrix
print("Classification Report:\n", classification_report(y_test, y_pred))
confusion_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion_mat)

# Visualize confusion matrix
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Save the cleaned dataset (optional)
df.to_csv(r'C:\Users\glori\Desktop\CSE805\CSE805\Graduate Seminar\cleanedCDK.csv', index=False)
print("Data cleaning completed. Cleaned data saved to 'cleanedCDK.csv'")

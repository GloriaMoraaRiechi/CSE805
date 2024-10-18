import pandas as pd  # used to load and manipulate data and for one-hot encoding
import numpy as np  # data manipulations
import matplotlib.pyplot as plt  # drawing graphs
from sklearn.utils import resample  # downsample the dataset
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier  # Added KNN import
from sklearn.model_selection import GridSearchCV  # will do cross-validation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import seaborn as sns
from scipy.stats import zscore
from sklearn.metrics import classification_report

# STEP 1: Import the Data
df = pd.read_csv(r"C:\Users\glori\Desktop\CSE805\CSE805\Support Vector Machines\CDK_Data.csv")
print(df.head())
print(df.columns)

# STEP 2: CLEAN THE DATA

# Drop irrelevant columns
df.drop('PatientID', axis=1, inplace=True)
print(df.head())

# 1. Initial Exploration
print(df.info())
print(df.describe())
print(df.isnull().sum())
print("Columns")
print(df.columns)
print(len(df.columns))  # To check the number of columns

# 2. Rename columns for clarity
df.columns = ['Age', 'Gender', 'Ethnicity', 'SocioeconomicStatus',
              'EducationLevel', 'BMI', 'Smoking', 'AlcoholConsumption',
              'PhysicalActivity', 'DietQuality', 'SleepQuality',
              'FamilyHistoryKidneyDisease', 'FamilyHistoryHypertension',
              'FamilyHistoryDiabetes', 'PreviousAcuteKidneyInjury',
              'UrinaryTractInfections', 'SystolicBP', 'DiastolicBP',
              'FastingBloodSugar', 'HbA1c', 'SerumCreatinine', 'BUNLevels',
              'GFR', 'ProteinInUrine', 'ACR', 'SerumElectrolytesSodium',
              'SerumElectrolytesPotassium', 'SerumElectrolytesCalcium',
              'SerumElectrolytesPhosphorus', 'HemoglobinLevels', 'CholesterolTotal',
              'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides',
              'ACEInhibitors', 'Diuretics', 'NSAIDsUse', 'Statins',
              'AntidiabeticMedications', 'Edema', 'FatigueLevels',
              'NauseaVomiting', 'MuscleCramps', 'Itching', 'QualityOfLifeScore',
              'HeavyMetalsExposure', 'OccupationalExposureChemicals',
              'WaterQuality', 'MedicalCheckupsFrequency',
              'MedicationAdherence', 'HealthLiteracy',
              'Diagnosis', 'DoctorInCharge']

# 3. Handle missing values
# First, replace '\t?' with NaN
df = df.replace('\t?', np.nan)

# Convert numeric columns to float
numeric_columns = ['Age', 'BMI', 'SystolicBP', 'DiastolicBP', 'FastingBloodSugar',
                   'HbA1c', 'SerumCreatinine', 'BUNLevels', 'GFR', 'ProteinInUrine',
                   'ACR', 'SerumElectrolytesSodium', 'SerumElectrolytesPotassium',
                   'SerumElectrolytesCalcium', 'SerumElectrolytesPhosphorus',
                   'HemoglobinLevels', 'CholesterolTotal', 'CholesterolLDL',
                   'CholesterolHDL', 'CholesterolTriglycerides', 'QualityOfLifeScore']

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Impute missing values
numeric_imputer = SimpleImputer(strategy='mean')
categorical_imputer = SimpleImputer(strategy='most_frequent')

# Categorical columns
categorical_columns = ['Gender', 'Ethnicity', 'SocioeconomicStatus',
                       'EducationLevel', 'Smoking', 'AlcoholConsumption',
                       'PhysicalActivity', 'DietQuality', 'SleepQuality',
                       'FamilyHistoryKidneyDisease', 'FamilyHistoryHypertension',
                       'FamilyHistoryDiabetes', 'PreviousAcuteKidneyInjury',
                       'UrinaryTractInfections', 'ACEInhibitors',
                       'Diuretics', 'NSAIDsUse', 'Statins',
                       'AntidiabeticMedications', 'Edema',
                       'FatigueLevels', 'NauseaVomiting',
                       'MuscleCramps', 'Itching',
                       'HeavyMetalsExposure', 'OccupationalExposureChemicals',
                       'WaterQuality', 'MedicalCheckupsFrequency',
                       'MedicationAdherence', 'HealthLiteracy',
                       'Diagnosis', 'DoctorInCharge']

df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])
df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])

# 4. Check for outliers
def plot_histograms(dataframe, features):
    fig, axes = plt.subplots(len(features) // 3 + 1, 3, figsize=(20, 5 * (len(features) // 3 + 1)))
    for i, feature in enumerate(features):
        sns.histplot(dataframe[feature], ax=axes[i // 3, i % 3], bins=30, kde=True)
        axes[i // 3, i % 3].set_title(feature)
        axes[i // 3, i % 3].set_xlabel('Value')
        axes[i // 3, i % 3].set_ylabel('Frequency')
    plt.tight_layout()
    plt.show()

# Call the histogram plotting function
plot_histograms(df, numeric_columns)

# Calculate Z-scores
z_scores = np.abs(zscore(df[numeric_columns]))

# Identify outliers based on Z-scores
outliers_z = (z_scores > 3).sum(axis=0)

# Print the features with outlier counts using Z-scores
for feature, count in zip(numeric_columns, outliers_z):
    print(f"{feature}: {count} outliers (Z-score method)")

# 5. Final check for missing values
print(df.isnull().sum())
print(df.dtypes)

# 6. Save cleaned dataset
df.to_csv(r'C:\Users\glori\Desktop\CSE805\CSE805\Support Vector Machines\cleaned_kidney_disease.csv', index=False)
print("Data cleaning completed. Cleaned data saved to 'cleaned_kidney_disease.csv'")

# STEP 3: NORMALIZATION
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# STEP 4: FEATURE SELECTION
# Assuming 'Diagnosis' is the target variable
X = df.drop('Diagnosis', axis=1)  # Features
y = df['Diagnosis']  # Target variable

# Optional: One-hot encoding for categorical variables
X = pd.get_dummies(X, drop_first=True)

# STEP 5: SPLIT THE DATASET INTO TRAINING AND TEST SETS
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 6: MODEL TRAINING WITH SVM
# Create SVM model
svm_model = SVC(kernel='linear')  # You can choose different kernels like 'rbf' or 'poly'

# Fit the model
svm_model.fit(X_train, y_train)

# STEP 7: MODEL TRAINING WITH KNN
# Create KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)  # You can tune n_neighbors

# Fit the model
knn_model.fit(X_train, y_train)

# STEP 8: MODEL EVALUATION
# Predict on test set using SVM
y_pred_svm = svm_model.predict(X_test)

# Predict on test set using KNN
y_pred_knn = knn_model.predict(X_test)

# Confusion Matrix for SVM
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
ConfusionMatrixDisplay(conf_matrix_svm, display_labels=svm_model.classes_).plot()
plt.title("Confusion Matrix for SVM")
plt.show()

# Confusion Matrix for KNN
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
ConfusionMatrixDisplay(conf_matrix_knn, display_labels=knn_model.classes_).plot()
plt.title("Confusion Matrix for KNN")
plt.show()

# STEP 9: PERFORMANCE METRICS
# SVM Classification Report
print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm))

# KNN Classification Report
print("KNN Classification Report:")
print(classification_report(y_test, y_pred_knn))

# STEP 10: PCA (Optional)
pca = PCA(n_components=2)  # Reduce to 2 components for visualization
X_pca = pca.fit_transform(X)

# Visualize PCA result
plt.figure(figsize=(10, 7))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.title("PCA of Kidney Disease Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()
plt.show()

print("Model training and evaluation completed.")

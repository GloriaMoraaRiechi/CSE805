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

from Regularization.lasso import scaler

# STEP1. Import the Data
df = pd.read_csv(r"C:\Users\glori\Desktop\CSE805\CSE805\Support Vector Machines\CDK_Data.csv")
print(df.head())
print(df.columns)
# STEP 2: CLEAN THE DATA#

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
# def plot_boxplot(dataframe, features):
#     fig, axes = plt.subplots(len(features) // 3 + 1, 3, figsize=(20, 5 * (len(features) // 3 + 1)))
#     for i, feature in enumerate(features):
#         sns.boxplot(x=dataframe[feature], ax=axes[i // 3, i % 3])
#     plt.tight_layout()
#     plt.show()
#
# plot_boxplot(df, numeric_columns)

# 4. Check for distributions using histograms
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

from scipy.stats import zscore

# Calculate Z-scores
z_scores = np.abs(zscore(df[numeric_columns]))

# Identify outliers based on Z-scores
outliers_z = (z_scores > 3).sum(axis=0)

# Print the features with outlier counts using Z-scores
for feature, count in zip(numeric_columns, outliers_z):
    print(f"{feature}: {count} outliers (Z-score method)")


# 5. Handle outliers
# Q1 = df['HbA1c'].quantile(0.25)
# Q3 = df['HbA1c'].quantile(0.75)
# IQR = Q3 - Q1
# lower_bound = Q1 - (1.5 * IQR)
# upper_bound = Q3 + (1.5 * IQR)
# df['HbA1c'] = df['HbA1c'].clip(lower_bound, upper_bound)

# 6. Final check
print(df.isnull().sum())
print(df.dtypes)

# 7. Save cleaned dataset
df.to_csv(r'C:\Users\glori\Desktop\CSE805\CSE805\Support Vector Machines\cleaned_kidney_disease.csv', index=False)

print("Data cleaning completed. Cleaned data saved to 'cleaned_kidney_disease.csv'")

scaler = StandardScaler()


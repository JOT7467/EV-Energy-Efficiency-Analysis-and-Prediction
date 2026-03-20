
#Importing Nympy and Pandas from Libraries
import numpy as np
import pandas as pd

#Creating Datasets

# Dataset 1: Patient Records
patients = pd.DataFrame({
    'patient_id': [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
    'age': [45, 62, 38, 55, 71, 49, 33, 58],
    'province': ['Nova Scotia', 'New Brunswick', 'Nova Scotia', 'Prince Edward Island',
                 'Newfoundland', 'Nova Scotia', 'New Brunswick', 'Nova Scotia'],
    'diagnosis': ['Diabetes', 'Hypertension', 'Diabetes', 'Asthma', 'Hypertension',
                  'Diabetes', 'Asthma', 'Hypertension'],
    'treatment_duration': [30, 45, 28, np.nan, 60, 35, np.nan, 52],
    'readmission': ['No', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'Yes']
})

# Dataset 2: Treatment Outcomes
outcomes = pd.DataFrame({
    'patient_id': [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
    'treatment_type': ['Medication', 'Lifestyle Change', 'Medication', 'Therapy',
                       'Medication', 'Lifestyle Change', 'Therapy', 'Medication'],
    'initial_blood_pressure': [145, 160, 152, np.nan, 168, 148, np.nan, 155],
    'final_blood_pressure': [128, 142, 135, np.nan, 149, 138, np.nan, 133],
    'patient_satisfaction': [8.5, 7.2, 9.0, 8.3, 6.8, 7.9, 9.2, 8.7]
})

#Part 1: NumPy Array Operations

#Age Statistics
age_array = patients['age'].to_numpy()

#Mean Age of Patients: It represents the average patient Age
The_Mean_Age = np.mean(age_array)

#Median Age of Patients: It represents the middle value when ages are ordered
The_Median_Age = np.median(age_array)

#Standard Deviation of Patient Records: It measures how much the patient ages vary around the mean
The_StandardDev_Age = np.std(age_array)

print("Mean Age:", The_Mean_Age)
print("Median age:", The_Median_Age)
print("Standard deviation:", The_StandardDev_Age)

# Explanation: The results show that the average patient is about 51 years old, with moderate variation of age across patients.

# 2D Numpy Array

td_Null_filled = patients['treatment_duration'].fillna(0)

array_2D = np.column_stack((
    patients['patient_id'],
    patients['age'],
    td_Null_filled
))

print("2D Array:\n", array_2D)
print("Shape:", array_2D.shape)

#Explanation: The treatment duration column contained two nulls. thia was replace with zero as displayed in the output.
#The array was structured with 3 columns representing Patient ID, Age and Treatment duration.
#This allows numerica operations on patient characteristics in matrix form.


#Boolean Mask for Patient more than 60 years:
mask = patients['age'] > 60     #To filter out patients who's age is greater than 60.
pat_mask = patients.loc[mask, 'patient_id'].to_numpy()
print("Patients over 60 IDs:", pat_mask)

#This demonstrates conditional selection using vectorized operation.
#Explanation: the output shows that only two patients have age greater than 60 years.


#Correlation between age and treatment duration

#Creating a subset dataset from the patients records droping all NA.
new_patient = patients.dropna(subset=['treatment_duration'])
print(new_patient)

corr = np.corrcoef(new_patient['age'], new_patient['treatment_duration'])[0,1]
print("Correlation coefficient:", corr)

#Explanation: The coefficient value of 0.95 indicates a very strong positive relationship between Age and Treatment Duration
# Older patients tend to have longer treatment durations.


#PART 2: Pandas Data Manipulation
#5
print("Patients head:\n", patients.head())
print("\nPatients info:")
patients.info()

print("\nOutcomes head:\n", outcomes.head())
print("\nOutcomes info:")
outcomes.info()

#Explanation: The head function shows the first 5 rows of the data
#The info() shows the data type of all the variables, Number of Non-Null values and dataset size
#This helps varify correct data loading and identify missing values


#6A : Count of Null Values

print("Patients NULL count:\n", patients.isnull().sum())
print("\nOutcomes NULL count:\n", outcomes.isnull().sum())

#Explanation: The results show that in the patient records, only treatment duration have two missing values
#and in the outcome records, initial blood pressure and final blood pressure have 2 missing values each.

#6B Filling Null Values in Treatment duration with the median value

patients['treatment_duration'] = patients.groupby('diagnosis')['treatment_duration']\
                                          .transform(lambda x: x.fillna(x.median()))

#6C : Drop rows with NULL blood pressure

outcomes_clean = outcomes.dropna(subset=['initial_blood_pressure', 'final_blood_pressure'])

#7: Merging Datasets

merged_Data = pd.merge(patients, outcomes_clean, on='patient_id', how='inner')
print("Merged DataFrame:\n", merged_Data)

#Explanation: The merged data contains only 6 patients which represent patients with complete records.
#An inner join was performed using patient_id. which means only patients present in both datasets after
#cleaning remained in the merged data. This ensure all variables needed for modeling are present.

# PART 3: Machine Learning Modeling with scikit

#8A : Feature matrix X

X = merged_Data[['age', 'treatment_duration',
            'initial_blood_pressure', 'patient_satisfaction']]

#8B. Target vector y

y = merged_Data['readmission'].map({'Yes':1, 'No':0})

#The readmission varaibles are converted to numeric form becasue machine learning models require numeric targets.
#Explanation: This means that we selected predicted variables based on clinical relevance.


#9: Train Test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Explanation: the datasets was divided as follows: 80% as training dataset and 20% as testing dataset.
#This is done so that the model learns from one portion and is evaluated on unseen data.
#This prevent overfiting and gives realistic performance estimate.


#10: StandardScaler normalization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaled feature means:", X_train_scaled.mean(axis=0))
print("Scaled feature std dev:", X_train_scaled.std(axis=0))

#Standardscaler standardizes variables so they are on the same scale
#This improves model performance because logistic regression is sensitive to featured magnitude
#Explanation: The result shows that the mean of each scaled feature is 0 and the standard deviation is 1.

#11 : Logistics regression Model

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)

print("Model accuracy:", accuracy)

#Explanation: The model accuracy is 1.0(100%). this means that the model correctly predicted readmission status for all patients in the test set
#Although the model achieved 100% accuracy, this result is likely due to the extremely small sample size rather
#than true predictive power. The model’s performance should not be considered reliable without testing on a much larger dataset.

#Recommendation:
#More dataset, Cross_Validation and Additional prediction would improve the reliability of the model.
#Thank You.








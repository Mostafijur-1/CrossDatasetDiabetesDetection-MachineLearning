import pandas as pd
import numpy as np

# Load the hospital dataset
df = pd.read_csv('datasets/diabetic_data.csv')

# Helper to get midpoint of age range
def age_midpoint(age_range):
    if pd.isnull(age_range):
        return np.nan
    age_range = age_range.strip('[]').replace(')', '')
    start, end = age_range.split('-')
    return (int(start) + int(end)) / 2

# Map max_glu_serum to numeric
def map_glucose(val):
    if val == 'None':
        return np.nan
    elif val == 'Norm':
        return 100
    elif val == '>200':
        return 220
    elif val == '>300':
        return 320
    else:
        return np.nan

# Map insulin usage to binary
def map_insulin(val):
    return 1 if val.lower() == 'up' or val.lower() == 'steady' or val.lower() == 'down' else 0

# Map readmitted to binary
def map_outcome(val):
    if val == 'NO':
        return 0
    else:
        return 1

# Gender-based pregnancies proxy
def map_pregnancies(row):
    if row['gender'] == 'Female':
        return row['number_inpatient']
    else:
        return 0

mapped = pd.DataFrame()
mapped['Pregnancies'] = df.apply(map_pregnancies, axis=1)
mapped['Glucose'] = df['max_glu_serum'].apply(map_glucose)
mapped['BloodPressure'] = df['time_in_hospital'] * 5 + 70
mapped['SkinThickness'] = df['num_procedures'] * 5
mapped['Insulin'] = df['insulin'].apply(map_insulin)
mapped['BMI'] = 20 + df['num_medications'] * 0.5
mapped['DiabetesPedigreeFunction'] = df['number_diagnoses'] * 0.1
mapped['Age'] = df['age'].apply(age_midpoint)
mapped['Outcome'] = df['readmitted'].apply(map_outcome)

mapped.to_csv('datasets/diabetic_data_mapped.csv', index=False)
print('Mapped CSV saved as datasets/diabetic_data_mapped.csv')

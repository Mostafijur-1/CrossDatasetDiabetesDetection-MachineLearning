import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def preprocess_dataset1():
    """Load and preprocess the Pima Indians diabetes dataset"""
    df = pd.read_csv('datasets/diabetes.csv')
    print(f"Dataset 1 shape: {df.shape}")
    print(f"Dataset 1 columns: {df.columns.tolist()}")
    return df

def preprocess_dataset2():
    """Load and preprocess the diabetic_data.csv to match features with dataset 1"""
    df = pd.read_csv('datasets/diabetic_data.csv')
    print(f"Original Dataset 2 shape: {df.shape}")
    
    # Create a new dataframe with mapped features
    processed_df = pd.DataFrame()
    
    # 1. Age mapping: Convert age ranges to numeric values (midpoint of range)
    age_mapping = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
        '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
        '[80-90)': 85, '[90-100)': 95
    }
    processed_df['Age'] = df['age'].map(age_mapping)
    
    # 2. Pregnancies: Use gender (Female = potential pregnancies, Male = 0)
    # For females, use number of inpatient visits as proxy for pregnancies
    processed_df['Pregnancies'] = np.where(
        df['gender'] == 'Female', 
        df['number_inpatient'].fillna(0),  # Use inpatient visits as proxy
        0  # Males have 0 pregnancies
    )
    
    # 3. Glucose: Map max_glu_serum to numeric values
    glucose_mapping = {
        'None': 100,    # Normal glucose
        'Norm': 100,    # Normal glucose  
        '>200': 250,    # High glucose
        '>300': 350     # Very high glucose
    }
    processed_df['Glucose'] = df['max_glu_serum'].map(glucose_mapping).fillna(100)
    
    # 4. BloodPressure: Use time_in_hospital as proxy (longer stay = higher BP risk)
    # Normalize to typical BP range
    processed_df['BloodPressure'] = 70 + (df['time_in_hospital'] * 5)  # 70-120 range
    
    # 5. SkinThickness: Use number of procedures as proxy (medical complexity)
    processed_df['SkinThickness'] = df['num_procedures'] * 5  # Scale to typical range
    
    # 6. Insulin: Map insulin medication usage
    processed_df['Insulin'] = np.where(
        df['insulin'].isin(['Up', 'Down', 'Steady']), 
        100,  # On insulin
        0     # Not on insulin
    )
    
    # 7. BMI: Use number of medications as proxy for health complexity
    # Scale to typical BMI range (18-40)
    processed_df['BMI'] = 20 + (df['num_medications'] * 0.5)
    
    # 8. DiabetesPedigreeFunction: Use number of diagnoses as proxy for genetic risk
    processed_df['DiabetesPedigreeFunction'] = df['number_diagnoses'] * 0.1
    
    # 9. Outcome: Map readmitted status to diabetes outcome
    # '>30' and '<30' (readmitted) = 1 (diabetes positive), 'NO' = 0
    processed_df['Outcome'] = np.where(
        df['readmitted'].isin(['>30', '<30']), 
        1,  # Readmitted = diabetes complications
        0   # Not readmitted
    )
    
    # Remove rows with missing values
    processed_df = processed_df.dropna()
    
    print(f"Processed Dataset 2 shape: {processed_df.shape}")
    print(f"Processed Dataset 2 columns: {processed_df.columns.tolist()}")
    
    return processed_df

def cross_dataset_training():
    """Train on one dataset and test on another"""
    
    # Load both datasets
    dataset1 = preprocess_dataset1()
    dataset2 = preprocess_dataset2()
    
    print("\n" + "="*50)
    print("CROSS-DATASET TRAINING ANALYSIS")
    print("="*50)
    
    # Prepare data for both datasets
    feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                      'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    X1 = dataset1[feature_columns]
    y1 = dataset1['Outcome']
    
    X2 = dataset2[feature_columns]
    y2 = dataset2['Outcome']
    
    # Standardize features
    scaler1 = StandardScaler()
    scaler2 = StandardScaler()
    
    X1_scaled = scaler1.fit_transform(X1)
    X2_scaled = scaler2.fit_transform(X2)
    
    # Split dataset 1 for training
    X1_train, X1_test, y1_train, y1_test = train_test_split(
        X1_scaled, y1, test_size=0.2, random_state=42
    )
    
    print(f"\nDataset 1 - Training samples: {len(X1_train)}, Test samples: {len(X1_test)}")
    print(f"Dataset 2 - Test samples: {len(X2_scaled)}")
    
    # Train model on dataset 1
    print("\n1. Training SVM model on Dataset 1 (Pima Indians)...")
    model1 = svm.SVC(kernel='linear', random_state=42)
    model1.fit(X1_train, y1_train)
    
    # Test on dataset 1 (same dataset)
    y1_pred = model1.predict(X1_test)
    accuracy1 = accuracy_score(y1_test, y1_pred)
    print(f"   Accuracy on Dataset 1 test set: {accuracy1:.4f}")
    
    # Test on dataset 2 (cross-dataset)
    print("\n2. Testing trained model on Dataset 2 (Hospital data)...")
    
    # We need to scale dataset 2 features using dataset 1's scaler for consistency
    X2_scaled_consistent = scaler1.transform(X2)  # Use same scaler as training
    y2_pred = model1.predict(X2_scaled_consistent)
    accuracy2 = accuracy_score(y2, y2_pred)
    print(f"   Cross-dataset accuracy: {accuracy2:.4f}")
    
    print(f"\n3. Classification Report for Cross-Dataset Testing:")
    print(classification_report(y2, y2_pred))
    
    # Train model on dataset 2 and test on dataset 1
    print("\n" + "-"*50)
    print("REVERSE: Training on Dataset 2, Testing on Dataset 1")
    print("-"*50)
    
    # Split dataset 2 for training
    X2_train, X2_test, y2_train, y2_test = train_test_split(
        X2_scaled, y2, test_size=0.2, random_state=42
    )
    
    print(f"Training SVM model on Dataset 2...")
    model2 = svm.SVC(kernel='linear', random_state=42)
    model2.fit(X2_train, y2_train)
    
    # Test on dataset 2
    y2_pred_same = model2.predict(X2_test)
    accuracy2_same = accuracy_score(y2_test, y2_pred_same)
    print(f"Accuracy on Dataset 2 test set: {accuracy2_same:.4f}")
    
    # Test on dataset 1 (cross-dataset)
    X1_scaled_consistent = scaler2.transform(X1)  # Use dataset 2's scaler
    y1_pred_cross = model2.predict(X1_scaled_consistent)
    accuracy1_cross = accuracy_score(y1, y1_pred_cross)
    print(f"Cross-dataset accuracy (Dataset 2 -> Dataset 1): {accuracy1_cross:.4f}")
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Dataset 1 -> Dataset 1: {accuracy1:.4f}")
    print(f"Dataset 1 -> Dataset 2: {accuracy2:.4f}")
    print(f"Dataset 2 -> Dataset 2: {accuracy2_same:.4f}")
    print(f"Dataset 2 -> Dataset 1: {accuracy1_cross:.4f}")
    
    # Feature importance analysis
    print(f"\nFeature Statistics:")
    print(f"Dataset 1 features mean: {X1.mean().round(2).to_dict()}")
    print(f"Dataset 2 features mean: {X2.mean().round(2).to_dict()}")
    
    return {
        'model1': model1,
        'model2': model2,
        'scaler1': scaler1,
        'scaler2': scaler2,
        'accuracies': {
            'dataset1_on_dataset1': accuracy1,
            'dataset1_on_dataset2': accuracy2,
            'dataset2_on_dataset2': accuracy2_same,
            'dataset2_on_dataset1': accuracy1_cross
        }
    }

if __name__ == "__main__":
    results = cross_dataset_training()

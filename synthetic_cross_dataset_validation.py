# -*- coding: utf-8 -*-
"""
Enhanced Cross-Dataset Validation with Synthetic Data Generation
Train on MulticlassDiabetesDataset and test on:
1. Mapped Pima dataset (direct mapping)
2. Synthetic dataset generated from Pima (using SMOTE and feature engineering)
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_multiclass_dataset():
    """Load and preprocess the MulticlassDiabetesDataset (training dataset)"""
    print("Loading MulticlassDiabetesDataset...")
    df = pd.read_csv('./datasets/MulticlassDiabetesDataset.csv')
    
    # Handle categorical columns
    if df['Gender'].dtype == 'object':
        df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    
    # Handle missing values
    imputer = KNNImputer(n_neighbors=5)
    df[df.columns] = imputer.fit_transform(df)
    
    # Feature engineering
    df['TG_HDL_Ratio'] = df['TG'] / (df['HDL'] + 1)
    df['BMI_Age'] = df['BMI'] * df['AGE']
    df['LDL_HDL_Ratio'] = df['LDL'] / (df['HDL'] + 1)
    
    # Split features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    print(f"MulticlassDiabetesDataset shape: {df.shape}")
    print(f"Features: {list(X.columns)}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y

def load_and_map_pima_dataset():
    """Load Pima dataset and map features to match MulticlassDiabetesDataset"""
    print("\nLoading and mapping Pima Indians Diabetes Dataset...")
    pima_df = pd.read_csv('./datasets/diabetes.csv')
    
    # Create a mapped dataframe with similar structure
    mapped_df = pd.DataFrame()
    
    # Direct mappings where possible
    mapped_df['AGE'] = pima_df['Age']
    mapped_df['BMI'] = pima_df['BMI']
    
    # Gender: Pima dataset is all females, so set to 1
    mapped_df['Gender'] = 1
    
    # Map glucose level (normalize to similar range as multiclass dataset)
    mapped_df['HbA1c'] = pima_df['Glucose'] / 40.0
    
    # Use available features as proxies for missing ones
    mapped_df['Chol'] = pima_df['DiabetesPedigreeFunction'] * 10 + 3
    mapped_df['Cr'] = pima_df['BloodPressure'] / 20.0
    mapped_df['Urea'] = pima_df['SkinThickness'] / 5.0
    
    # Create synthetic lipid profile based on BMI and age
    mapped_df['TG'] = (mapped_df['BMI'] / 10) + (mapped_df['AGE'] / 50)
    mapped_df['HDL'] = 2.5 - (mapped_df['BMI'] / 50)
    mapped_df['LDL'] = (mapped_df['BMI'] / 15) + 1
    mapped_df['VLDL'] = mapped_df['TG'] / 5
    
    # Handle missing values and ensure positive values
    mapped_df = mapped_df.fillna(mapped_df.mean())
    mapped_df = mapped_df.abs()
    
    # Feature engineering (same as training dataset)
    mapped_df['TG_HDL_Ratio'] = mapped_df['TG'] / (mapped_df['HDL'] + 1)
    mapped_df['BMI_Age'] = mapped_df['BMI'] * mapped_df['AGE']
    mapped_df['LDL_HDL_Ratio'] = mapped_df['LDL'] / (mapped_df['HDL'] + 1)
    
    # Target variable (binary classification: 0 = No diabetes, 1 = Diabetes)
    y_pima = pima_df['Outcome']
    
    print(f"Pima dataset shape: {pima_df.shape}")
    print(f"Mapped Pima shape: {mapped_df.shape}")
    print(f"Pima target distribution: {y_pima.value_counts().to_dict()}")
    
    return mapped_df, y_pima

def generate_synthetic_dataset_from_pima(X_mapped, y_pima, target_samples=500):
    """Generate synthetic dataset from mapped Pima data using SMOTE and feature engineering"""
    print(f"\nGenerating synthetic dataset from Pima data...")
    
    # First, use SMOTE to balance and expand the dataset
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_resampled, y_resampled = smote.fit_resample(X_mapped, y_pima)
    
    print(f"After SMOTE - Shape: {X_resampled.shape}, Target distribution: {pd.Series(y_resampled).value_counts().to_dict()}")
    
    # Add noise and variations to create more realistic synthetic data
    np.random.seed(42)
    
    # Create variations by adding controlled noise
    X_synthetic = X_resampled.copy()
    
    # Add small random variations to continuous features
    continuous_features = ['AGE', 'BMI', 'HbA1c', 'Chol', 'Cr', 'Urea', 'TG', 'HDL', 'LDL', 'VLDL']
    for feature in continuous_features:
        if feature in X_synthetic.columns:
            # Add 5% random noise
            noise = np.random.normal(0, X_synthetic[feature].std() * 0.05, len(X_synthetic))
            X_synthetic[feature] = X_synthetic[feature] + noise
            # Ensure positive values
            X_synthetic[feature] = np.abs(X_synthetic[feature])
    
    # Create multiclass labels based on severity
    y_synthetic = create_multiclass_labels(X_synthetic, y_resampled)
    
    # Limit to target number of samples
    if len(X_synthetic) > target_samples:
        indices = np.random.choice(len(X_synthetic), target_samples, replace=False)
        X_synthetic = X_synthetic.iloc[indices].reset_index(drop=True)
        y_synthetic = y_synthetic.iloc[indices].reset_index(drop=True)
    
    print(f"Synthetic dataset shape: {X_synthetic.shape}")
    print(f"Synthetic target distribution: {y_synthetic.value_counts().to_dict()}")
    
    return X_synthetic, y_synthetic

def create_multiclass_labels(X_data, y_binary):
    """Convert binary labels to multiclass based on feature values"""
    y_multi = pd.Series(y_binary.copy())
    
    # Create conditions for different diabetes severity levels
    for i in range(len(y_multi)):
        if y_binary[i] == 0:  # No diabetes
            y_multi.iloc[i] = 0
        else:  # Has diabetes - classify severity
            # Use HbA1c, BMI, and age to determine severity
            hba1c = X_data.iloc[i]['HbA1c'] if 'HbA1c' in X_data.columns else 0
            bmi = X_data.iloc[i]['BMI'] if 'BMI' in X_data.columns else 0
            age = X_data.iloc[i]['AGE'] if 'AGE' in X_data.columns else 0
            
            # Severity score based on risk factors
            severity_score = 0
            if hba1c > 3.5:  # High HbA1c
                severity_score += 1
            if bmi > 30:  # Obesity
                severity_score += 1
            if age > 50:  # Older age
                severity_score += 1
            
            # Assign class based on severity
            if severity_score >= 2:
                y_multi.iloc[i] = 2  # Severe diabetes
            else:
                y_multi.iloc[i] = 1  # Moderate diabetes
    
    return y_multi

def train_models(X_train, y_train):
    """Train multiple models with hyperparameter tuning"""
    print("\nTraining models...")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Logistic Regression
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr_params = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['lbfgs']
    }
    lr_grid = GridSearchCV(lr, lr_params, cv=3, scoring='accuracy', n_jobs=-1)
    lr_grid.fit(X_train_scaled, y_train)
    lr_best = lr_grid.best_estimator_
    
    # Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(random_state=42)
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'min_samples_split': [2, 5]
    }
    rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='accuracy')
    rf_grid.fit(X_train_scaled, y_train)
    rf_best = rf_grid.best_estimator_
    
    # XGBoost
    print("Training XGBoost...")
    xgb = XGBClassifier(eval_metric='logloss', random_state=42)
    xgb_params = {
        'n_estimators': [100, 200],
        'max_depth': [3, 4],
        'learning_rate': [0.05, 0.1]
    }
    xgb_grid = GridSearchCV(xgb, xgb_params, cv=3, scoring='accuracy')
    xgb_grid.fit(X_train_scaled, y_train)
    xgb_best = xgb_grid.best_estimator_
    
    # Voting Ensemble
    print("Training Voting Ensemble...")
    voting_clf = VotingClassifier(
        estimators=[('rf', rf_best), ('xgb', xgb_best), ('lr', lr_best)],
        voting='soft',
        weights=[2, 3, 1]
    )
    voting_clf.fit(X_train_scaled, y_train)
    
    return {
        'scaler': scaler,
        'lr': lr_best,
        'rf': rf_best,
        'xgb': xgb_best,
        'ensemble': voting_clf
    }

def evaluate_models(models, X_test, y_test, dataset_name):
    """Evaluate trained models on test dataset"""
    print(f"\n{'='*60}")
    print(f"{dataset_name.upper()} VALIDATION RESULTS")
    print("="*60)
    
    # Scale test data using the same scaler
    X_test_scaled = models['scaler'].transform(X_test)
    
    results = {}
    
    # Evaluate each model
    for model_name, model in models.items():
        if model_name == 'scaler':
            continue
            
        print(f"\n{model_name.upper()} Results:")
        print("-" * 40)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        results[model_name] = accuracy
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix:")
        print(cm)
        print()
    
    return results

def main():
    """Main execution function"""
    print("="*80)
    print("ENHANCED CROSS-DATASET DIABETES PREDICTION VALIDATION")
    print("Training on: MulticlassDiabetesDataset")
    print("Testing on: 1) Mapped Pima Dataset  2) Synthetic Dataset from Pima")
    print("="*80)
    
    # Load training dataset (MulticlassDiabetesDataset)
    X_train, y_train = load_and_preprocess_multiclass_dataset()
    
    # Load and map test dataset (Pima)
    X_pima_mapped, y_pima = load_and_map_pima_dataset()
    
    # Generate synthetic dataset from Pima
    X_synthetic, y_synthetic = generate_synthetic_dataset_from_pima(X_pima_mapped, y_pima)
    
    # Ensure all datasets have the same features
    X_pima_mapped = X_pima_mapped[X_train.columns]
    X_synthetic = X_synthetic[X_train.columns]
    
    print(f"\nDataset Shapes:")
    print(f"Training (MulticlassDiabetes): {X_train.shape}")
    print(f"Test 1 (Mapped Pima): {X_pima_mapped.shape}")
    print(f"Test 2 (Synthetic): {X_synthetic.shape}")
    
    # Train models
    models = train_models(X_train, y_train)
    
    # Evaluate on training dataset (sanity check)
    print(f"\n{'='*60}")
    print("SANITY CHECK: Training Dataset Performance")
    print("="*60)
    X_train_scaled = models['scaler'].transform(X_train)
    train_pred = models['ensemble'].predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, train_pred)
    print(f"Training Accuracy (Ensemble): {train_accuracy:.4f}")
    
    # Cross-dataset evaluation on mapped Pima
    pima_results = evaluate_models(models, X_pima_mapped, y_pima, "Mapped Pima Dataset")
    
    # Cross-dataset evaluation on synthetic dataset
    synthetic_results = evaluate_models(models, X_synthetic, y_synthetic, "Synthetic Dataset")
    
    # Summary
    print(f"\n{'='*80}")
    print("COMPREHENSIVE SUMMARY")
    print("="*80)
    print(f"Training Dataset: MulticlassDiabetesDataset ({X_train.shape[0]} samples)")
    print(f"Test Dataset 1: Mapped Pima Dataset ({X_pima_mapped.shape[0]} samples)")
    print(f"Test Dataset 2: Synthetic Dataset ({X_synthetic.shape[0]} samples)")
    
    print(f"\nMapped Pima Dataset Performance:")
    for model_name, accuracy in pima_results.items():
        print(f"  {model_name.capitalize()}: {accuracy:.4f}")
    
    print(f"\nSynthetic Dataset Performance:")
    for model_name, accuracy in synthetic_results.items():
        print(f"  {model_name.capitalize()}: {accuracy:.4f}")
    
    print(f"\nKey Insights:")
    print(f"- Training accuracy: {train_accuracy:.4f} (expected to be high)")
    print(f"- Cross-dataset validation shows model generalization challenges")
    print(f"- Synthetic data may provide better feature alignment")
    print(f"- This demonstrates real-world ML deployment challenges")

if __name__ == "__main__":
    main()

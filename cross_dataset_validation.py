# -*- coding: utf-8 -*-
"""
Cross-Dataset Validation Script
Train on MulticlassDiabetesDataset and test on Pima Indians Diabetes Dataset
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
    # Pima glucose is typically 70-200, normalize to similar range as HbA1c
    mapped_df['HbA1c'] = pima_df['Glucose'] / 40.0  # Rough conversion
    
    # Use available features as proxies for missing ones
    # Use DiabetesPedigreeFunction as a proxy for genetic/hereditary factors
    mapped_df['Chol'] = pima_df['DiabetesPedigreeFunction'] * 10 + 3  # Scale to reasonable cholesterol range
    
    # Use SkinThickness and BloodPressure as proxies
    mapped_df['Cr'] = pima_df['BloodPressure'] / 20.0  # Scale blood pressure to creatinine-like range
    mapped_df['Urea'] = pima_df['SkinThickness'] / 5.0  # Scale skin thickness to urea-like range
    
    # Create synthetic lipid profile based on BMI and age
    mapped_df['TG'] = (mapped_df['BMI'] / 10) + (mapped_df['AGE'] / 50)  # Triglycerides
    mapped_df['HDL'] = 2.5 - (mapped_df['BMI'] / 50)  # HDL (inverse relation with BMI)
    mapped_df['LDL'] = (mapped_df['BMI'] / 15) + 1  # LDL
    mapped_df['VLDL'] = mapped_df['TG'] / 5  # VLDL calculation
    
    # Handle missing values and ensure positive values
    mapped_df = mapped_df.fillna(mapped_df.mean())
    mapped_df = mapped_df.abs()  # Ensure all values are positive
    
    # Feature engineering (same as training dataset)
    mapped_df['TG_HDL_Ratio'] = mapped_df['TG'] / (mapped_df['HDL'] + 1)
    mapped_df['BMI_Age'] = mapped_df['BMI'] * mapped_df['AGE']
    mapped_df['LDL_HDL_Ratio'] = mapped_df['LDL'] / (mapped_df['HDL'] + 1)
    
    # Target variable (binary classification: 0 = No diabetes, 1 = Diabetes)
    y_pima = pima_df['Outcome']
    
    print(f"Pima dataset shape: {pima_df.shape}")
    print(f"Mapped Pima shape: {mapped_df.shape}")
    print(f"Mapped features: {list(mapped_df.columns)}")
    print(f"Pima target distribution: {y_pima.value_counts().to_dict()}")
    
    return mapped_df, y_pima

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
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['lbfgs']
    }
    lr_grid = GridSearchCV(lr, lr_params, cv=5, scoring='accuracy', n_jobs=-1)
    lr_grid.fit(X_train_scaled, y_train)
    lr_best = lr_grid.best_estimator_
    
    # Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(random_state=42)
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
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

def evaluate_cross_dataset(models, X_test, y_test):
    """Evaluate trained models on cross-dataset (Pima)"""
    print("\n" + "="*50)
    print("CROSS-DATASET VALIDATION RESULTS")
    print("="*50)
    
    # Scale test data using the same scaler
    X_test_scaled = models['scaler'].transform(X_test)
    
    results = {}
    
    # Evaluate each model
    for model_name, model in models.items():
        if model_name == 'scaler':
            continue
            
        print(f"\n{model_name.upper()} Results:")
        print("-" * 30)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        results[model_name] = accuracy
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix:")
        print(cm)
    
    return results

def main():
    """Main execution function"""
    print("="*60)
    print("CROSS-DATASET DIABETES PREDICTION VALIDATION")
    print("Training on: MulticlassDiabetesDataset")
    print("Testing on: Pima Indians Diabetes Dataset")
    print("="*60)
    
    # Load training dataset (MulticlassDiabetesDataset)
    X_train, y_train = load_and_preprocess_multiclass_dataset()
    
    # Load and map test dataset (Pima)
    X_test, y_test = load_and_map_pima_dataset()
    
    # Ensure both datasets have the same features
    print(f"\nTraining features: {list(X_train.columns)}")
    print(f"Test features: {list(X_test.columns)}")
    
    # Reorder test features to match training features
    X_test = X_test[X_train.columns]
    
    print(f"\nFeature alignment successful!")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Train models
    models = train_models(X_train, y_train)
    
    # Evaluate on same dataset (sanity check)
    print(f"\n{'='*50}")
    print("SANITY CHECK: Training Dataset Performance")
    print("="*50)
    X_train_scaled = models['scaler'].transform(X_train)
    train_pred = models['ensemble'].predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, train_pred)
    print(f"Training Accuracy (Ensemble): {train_accuracy:.4f}")
    
    # Cross-dataset evaluation
    cross_results = evaluate_cross_dataset(models, X_test, y_test)
    
    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print("="*50)
    print(f"Training Dataset: MulticlassDiabetesDataset ({X_train.shape[0]} samples)")
    print(f"Test Dataset: Pima Indians Dataset ({X_test.shape[0]} samples)")
    print(f"\nCross-Dataset Performance:")
    for model_name, accuracy in cross_results.items():
        print(f"  {model_name.capitalize()}: {accuracy:.4f}")
    
    print(f"\nNote: Lower accuracy is expected in cross-dataset validation")
    print(f"due to domain shift between different datasets.")

if __name__ == "__main__":
    main()

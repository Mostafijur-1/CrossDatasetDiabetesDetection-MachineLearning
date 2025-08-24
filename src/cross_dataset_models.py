import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
import joblib
from cross_dataset_analysis import preprocess_dataset1, preprocess_dataset2

def train_cross_dataset_models():
    """Train models on both datasets and save them for comparison"""
    
    # Load and preprocess both datasets
    dataset1 = preprocess_dataset1()
    dataset2 = preprocess_dataset2()
    
    feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                      'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    # Prepare Dataset 1 (Pima)
    X1 = dataset1[feature_columns]
    y1 = dataset1['Outcome']
    
    scaler1 = StandardScaler()
    X1_scaled = scaler1.fit_transform(X1)
    
    # Train model on Dataset 1
    model1 = svm.SVC(kernel='linear', random_state=42)
    model1.fit(X1_scaled, y1)
    
    # Prepare Dataset 2 (Hospital)
    X2 = dataset2[feature_columns]
    y2 = dataset2['Outcome']
    
    scaler2 = StandardScaler()
    X2_scaled = scaler2.fit_transform(X2)
    
    # Train model on Dataset 2
    model2 = svm.SVC(kernel='linear', random_state=42)
    model2.fit(X2_scaled, y2)
    
    # Save models and scalers
    joblib.dump(model1, 'svm_diabetes_model_pima.pkl')
    joblib.dump(scaler1, 'scaler_pima.pkl')
    joblib.dump(model2, 'svm_diabetes_model_hospital.pkl')
    joblib.dump(scaler2, 'scaler_hospital.pkl')
    
    print("Models saved:")
    print("- svm_diabetes_model_pima.pkl (trained on Pima dataset)")
    print("- scaler_pima.pkl")
    print("- svm_diabetes_model_hospital.pkl (trained on Hospital dataset)")
    print("- scaler_hospital.pkl")
    
    # Cross-validation results
    X1_cross = scaler2.transform(X1)  # Scale Dataset 1 with Dataset 2 scaler
    X2_cross = scaler1.transform(X2)  # Scale Dataset 2 with Dataset 1 scaler
    
    pima_on_hospital = accuracy_score(y2, model1.predict(X2_cross))
    hospital_on_pima = accuracy_score(y1, model2.predict(X1_cross))
    
    results = {
        'pima_accuracy': accuracy_score(y1, model1.predict(X1_scaled)),
        'hospital_accuracy': accuracy_score(y2, model2.predict(X2_scaled)),
        'cross_pima_on_hospital': pima_on_hospital,
        'cross_hospital_on_pima': hospital_on_pima
    }
    
    return results

def predict_with_both_models(input_data):
    """Make predictions using both models and return comparison"""
    
    # Load models and scalers
    model_pima = joblib.load('svm_diabetes_model_pima.pkl')
    scaler_pima = joblib.load('scaler_pima.pkl')
    model_hospital = joblib.load('svm_diabetes_model_hospital.pkl')
    scaler_hospital = joblib.load('scaler_hospital.pkl')
    
    # Prepare input data
    input_array = np.array(input_data).reshape(1, -1)
    
    # Predict with Pima model
    input_scaled_pima = scaler_pima.transform(input_array)
    prediction_pima = model_pima.predict(input_scaled_pima)[0]
    confidence_pima = model_pima.decision_function(input_scaled_pima)[0]
    
    # Predict with Hospital model
    input_scaled_hospital = scaler_hospital.transform(input_array)
    prediction_hospital = model_hospital.predict(input_scaled_hospital)[0]
    confidence_hospital = model_hospital.decision_function(input_scaled_hospital)[0]
    
    results = {
        'pima_model': {
            'prediction': 'Diabetes Positive 🩺' if prediction_pima == 1 else 'Diabetes Negative ✅',
            'confidence': confidence_pima,
            'binary': prediction_pima
        },
        'hospital_model': {
            'prediction': 'Diabetes Positive 🩺' if prediction_hospital == 1 else 'Diabetes Negative ✅',
            'confidence': confidence_hospital,
            'binary': prediction_hospital
        },
        'consensus': {
            'agreement': prediction_pima == prediction_hospital,
            'final_prediction': 'Diabetes Positive 🩺' if (prediction_pima + prediction_hospital) >= 1 else 'Diabetes Negative ✅'
        }
    }
    
    return results

if __name__ == "__main__":
    print("Training cross-dataset models...")
    results = train_cross_dataset_models()
    
    print(f"\nModel Performance Summary:")
    print(f"Pima model accuracy: {results['pima_accuracy']:.4f}")
    print(f"Hospital model accuracy: {results['hospital_accuracy']:.4f}")
    print(f"Cross-validation (Pima -> Hospital): {results['cross_pima_on_hospital']:.4f}")
    print(f"Cross-validation (Hospital -> Pima): {results['cross_hospital_on_pima']:.4f}")
    
    # Test with sample data
    print(f"\nTesting with sample data:")
    sample_input = [6, 148, 72, 35, 0, 33.6, 0.627, 50]  # From your original dataset
    predictions = predict_with_both_models(sample_input)
    
    print(f"Input: {sample_input}")
    print(f"Pima Model: {predictions['pima_model']['prediction']}")
    print(f"Hospital Model: {predictions['hospital_model']['prediction']}")
    print(f"Models Agree: {predictions['consensus']['agreement']}")
    print(f"Final Consensus: {predictions['consensus']['final_prediction']}")

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load models and scalers
def load_models():
    """Load all available models and scalers"""
    models = {}
    
    # Original model (if exists)
    if os.path.exists("svm_diabetes_model.pkl"):
        models['original'] = {
            'model': joblib.load("svm_diabetes_model.pkl"),
            'scaler': joblib.load("scaler.pkl")
        }
    
    # Pima dataset model
    if os.path.exists("svm_diabetes_model_pima.pkl"):
        models['pima'] = {
            'model': joblib.load("svm_diabetes_model_pima.pkl"),
            'scaler': joblib.load("scaler_pima.pkl")
        }
    
    # Hospital dataset model
    if os.path.exists("svm_diabetes_model_hospital.pkl"):
        models['hospital'] = {
            'model': joblib.load("svm_diabetes_model_hospital.pkl"),
            'scaler': joblib.load("scaler_hospital.pkl")
        }
    
    return models

models = load_models()

@app.route("/")
def home():
    return render_template("index.html")

def predict_with_multiple_models(input_data):
    """Make predictions using all available models"""
    results = {}
    predictions = []
    confidences = []
    
    input_array = np.array(input_data).reshape(1, -1)
    
    for model_name, model_data in models.items():
        try:
            # Scale input data
            scaled_input = model_data['scaler'].transform(input_array)
            
            # Make prediction
            prediction = model_data['model'].predict(scaled_input)[0]
            confidence = model_data['model'].decision_function(scaled_input)[0]
            
            results[model_name] = {
                'prediction': prediction,
                'confidence': confidence,
                'result_text': "Diabetes Positive 🩺" if prediction == 1 else "Diabetes Negative ✅"
            }
            
            predictions.append(prediction)
            confidences.append(abs(confidence))
            
        except Exception as e:
            results[model_name] = {
                'error': str(e)
            }
    
    # Calculate consensus
    if predictions:
        # Weighted consensus based on confidence
        weighted_prediction = sum(p * c for p, c in zip(predictions, confidences)) / sum(confidences)
        consensus_prediction = 1 if weighted_prediction > 0.5 else 0
        consensus_confidence = np.mean(confidences)
        
        results['consensus'] = {
            'prediction': consensus_prediction,
            'confidence': consensus_confidence,
            'result_text': "Diabetes Positive 🩺" if consensus_prediction == 1 else "Diabetes Negative ✅",
            'agreement_rate': sum(1 for p in predictions if p == consensus_prediction) / len(predictions)
        }
    
    return results

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if request.is_json:
            req_data = request.get_json()
            data = [
                float(req_data["Pregnancies"]),
                float(req_data["Glucose"]),
                float(req_data["BloodPressure"]),
                float(req_data["SkinThickness"]),
                float(req_data["Insulin"]),
                float(req_data["BMI"]),
                float(req_data["DiabetesPedigreeFunction"]),
                float(req_data["Age"]),
            ]
            
            # Get predictions from all models
            all_predictions = predict_with_multiple_models(data)
            
            # Return detailed results
            response = {
                "predictions": all_predictions,
                "primary_result": all_predictions.get('consensus', {}).get('result_text', 
                                all_predictions.get('pima', {}).get('result_text',
                                all_predictions.get('original', {}).get('result_text', "Error")))
            }
            
            return jsonify(response)
            
        else:
            # Form submission
            data = [
                float(request.form["Pregnancies"]),
                float(request.form["Glucose"]),
                float(request.form["BloodPressure"]),
                float(request.form["SkinThickness"]),
                float(request.form["Insulin"]),
                float(request.form["BMI"]),
                float(request.form["DiabetesPedigreeFunction"]),
                float(request.form["Age"]),
            ]
            
            # Get predictions from all models
            all_predictions = predict_with_multiple_models(data)
            
            # Use consensus result if available, otherwise use best available model
            primary_result = all_predictions.get('consensus', {}).get('result_text', 
                            all_predictions.get('pima', {}).get('result_text',
                            all_predictions.get('original', {}).get('result_text', "Error")))
            
            return render_template("index.html", 
                                 prediction=primary_result,
                                 detailed_predictions=all_predictions)
                                 
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/models/info")
def models_info():
    """Endpoint to get information about loaded models"""
    model_info = {}
    for model_name in models.keys():
        model_info[model_name] = {
            'loaded': True,
            'type': 'SVM'
        }
    return jsonify(model_info)

@app.route("/predict/detailed", methods=["POST"])
def predict_detailed():
    """Endpoint for detailed predictions from all models"""
    try:
        if request.is_json:
            req_data = request.get_json()
        else:
            req_data = request.form.to_dict()
            
        data = [
            float(req_data["Pregnancies"]),
            float(req_data["Glucose"]),
            float(req_data["BloodPressure"]),
            float(req_data["SkinThickness"]),
            float(req_data["Insulin"]),
            float(req_data["BMI"]),
            float(req_data["DiabetesPedigreeFunction"]),
            float(req_data["Age"]),
        ]
        
        all_predictions = predict_with_multiple_models(data)
        
        return jsonify({
            "input_data": data,
            "predictions": all_predictions,
            "available_models": list(models.keys()),
            "consensus_available": 'consensus' in all_predictions
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    print(f"Loaded models: {list(models.keys())}")
    app.run(debug=True)

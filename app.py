from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("svm_diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")  # serves the HTML form

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
            np_array = np.asarray(data).reshape(1, -1)
            std_data = scaler.transform(np_array)
            prediction = model.predict(std_data)
            result = "Diabetes Positive 🩺" if prediction[0] == 1 else "Diabetes Negative ✅"
            return jsonify({"prediction": result})
        else:
            # fallback for form submission
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
            np_array = np.asarray(data).reshape(1, -1)
            std_data = scaler.transform(np_array)
            prediction = model.predict(std_data)
            result = "Diabetes Positive 🩺" if prediction[0] == 1 else "Diabetes Negative ✅"
            return render_template("index.html", prediction=result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)

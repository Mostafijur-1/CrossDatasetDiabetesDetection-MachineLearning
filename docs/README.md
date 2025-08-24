# CrossDatasetDiabetesDetection-MachineLearning

## Overview
This project is a Flask web application for predicting diabetes risk using a trained SVM model. Users can enter medical parameters via a form or paste comma-separated values to auto-fill the form and get instant predictions.

## Features
- Modern, desktop-oriented UI
- Predict diabetes risk using machine learning
- Two input methods: form fields or comma-separated values
- Results displayed instantly without page reload

## Getting Started

### Prerequisites
- Python 3.7+
- pip

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Mostafijur-1/CrossDatasetDiabetesDetection-MachineLearning.git
   cd CrossDatasetDiabetesDetection-MachineLearning
   ```
3. Ensure `svm_diabetes_model.pkl` and `scaler.pkl` are present in the project directory.

### Running the App
1. Start the Flask server:
   ```
   python app.py
   ```
2. Open your browser and go to `http://localhost:5000`

## How to Predict

### Method 1: Form Input
- Fill in each medical parameter in the form fields.
- Click **Predict Diabetes**.
- The result will appear below the form.





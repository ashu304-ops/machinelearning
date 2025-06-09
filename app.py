# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')
features = joblib.load('features.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Build feature vector in correct order
        input_data = [data[feature] for feature in features]
        input_scaled = scaler.transform([input_data])
        prediction = model.predict(input_scaled)[0]

        result = "likely diabetic" if prediction == 1 else "unlikely diabetic"
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)



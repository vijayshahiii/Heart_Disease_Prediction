from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)

with open('heart_disease_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = np.array([[ 
        data['age'], data['sex'], data['cp'], data['trestbps'], data['chol'],
        data['fbs'], data['restecg'], data['thalach'], data['exang'], 
        data['oldpeak'], data['slope'], data['ca'], data['thal']
    ]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    confidence = model.predict_proba(scaled_data).max()
    return jsonify({'prediction': int(prediction[0]), 'confidence': float(confidence)})

if __name__ == "__main__":
    app.run(debug=True)

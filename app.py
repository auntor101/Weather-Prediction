from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Load the best Random Forest model and Scaler
model = joblib.load('random_forest_model.joblib')
scaler = joblib.load('scaler.joblib')

# Define Weather Labels
weather_label_map = {0: 'Rain', 1: 'High Temperature', 2: 'Normal Temperature', 3: 'Cold'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        date_val = request.form['date']
        rainfall = float(request.form['rainfall'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        sunshine = float(request.form['sunshine'])

        # Convert date to ordinal format
        date_obj = datetime.strptime(date_val, '%Y-%m-%d')
        date_ordinal = date_obj.toordinal()

        # Prepare input for prediction
        input_data = np.array([[rainfall, temperature, humidity, sunshine, date_ordinal]])
        
        # Scale the input
        input_scaled = scaler.transform(input_data)

        # Predict weather category
        prediction = model.predict(input_scaled)
        predicted_label = weather_label_map.get(prediction[0], "Unknown")

        return render_template('index.html', prediction_text=f'Predicted Weather: {predicted_label}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

# API Endpoint for JSON Requests
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()

        # Extract inputs
        date_val = data['date']
        rainfall = float(data['rainfall'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        sunshine = float(data['sunshine'])

        # Convert date to ordinal format
        date_obj = datetime.strptime(date_val, '%Y-%m-%d')
        date_ordinal = date_obj.toordinal()

        # Prepare input for prediction
        input_data = np.array([[rainfall, temperature, humidity, sunshine, date_ordinal]])

        # Scale the input
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_scaled)
        predicted_label = weather_label_map.get(prediction[0], "Unknown")

        return jsonify({'prediction': predicted_label})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from datetime import datetime
import logging

# Initialize Flask app
app = Flask(__name__)

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the Random Forest model and Scaler
try:
    model = joblib.load('random_forest_model.joblib')
    scaler = joblib.load('scaler.joblib')
    logging.info("Model and scaler loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model or scaler: {e}")
    raise

# Define Weather Labels
weather_label_map = {0: 'Rain', 1: 'High Temperature', 2: 'Normal Temperature', 3: 'Cold'}

# Utility function to validate and preprocess inputs
def validate_and_prepare_inputs(date_val, rainfall, temperature, humidity, sunshine):
    try:
        # Validate numerical inputs
        rainfall = float(rainfall)
        temperature = float(temperature)
        humidity = float(humidity)
        sunshine = float(sunshine)

        if rainfall < 0 or temperature < -50 or temperature > 60 or humidity < 0 or humidity > 100 or sunshine < 0:
            raise ValueError("Input values are out of realistic ranges.")

        # Convert date to ordinal format
        date_obj = datetime.strptime(date_val, '%Y-%m-%d')
        date_ordinal = date_obj.toordinal()

        # Prepare input array
        input_data = np.array([[rainfall, temperature, humidity, sunshine, date_ordinal]])
        return input_data
    except ValueError as ve:
        logging.error(f"Validation error: {ve}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in input validation: {e}")
        raise

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        date_val = request.form['date']
        rainfall = request.form['rainfall']
        temperature = request.form['temperature']
        humidity = request.form['humidity']
        sunshine = request.form['sunshine']

        # Validate and prepare inputs
        input_data = validate_and_prepare_inputs(date_val, rainfall, temperature, humidity, sunshine)

        # Scale the input
        input_scaled = scaler.transform(input_data)

        # Predict weather category
        prediction = model.predict(input_scaled)
        predicted_label = weather_label_map.get(prediction[0], "Unknown")

        return render_template('index.html', prediction_text=f'Predicted Weather: {predicted_label}')

    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return render_template('index.html', prediction_text=f'Error: {e}')

# API endpoint for JSON requests
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()

        # Extract inputs
        date_val = data['date']
        rainfall = data['rainfall']
        temperature = data['temperature']
        humidity = data['humidity']
        sunshine = data['sunshine']

        # Validate and prepare inputs
        input_data = validate_and_prepare_inputs(date_val, rainfall, temperature, humidity, sunshine)

        # Scale the input
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_scaled)
        predicted_label = weather_label_map.get(prediction[0], "Unknown")

        return jsonify({'prediction': predicted_label})

    except Exception as e:
        logging.error(f"Error in API prediction: {e}")
        return jsonify({'error': str(e)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

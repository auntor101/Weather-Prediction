Weather Prediction for Chittagong

This machine learning-based weather prediction system uses historical meteorological data to forecast weather conditions in Chittagong, Bangladesh. The project applies advanced data preprocessing, hyperparameter tuning, and ensemble learning to enhance prediction accuracy and generalization. A Flask web application with a structured Bootstrap UI allows users to input weather parameters and obtain real-time predictions.

Introduction

Accurate weather forecasting is essential for agriculture, transportation, disaster management, and daily planning. This project builds a predictive model trained on historical weather data, integrating RandomizedSearchCV, SMOTE, and a stacking ensemble of multiple classifiers to improve accuracy and reduce overfitting.

The web interface provides a structured user experience and enables real-time predictions with probability scores for different weather conditions.

Dataset

The dataset is obtained from Mendeley Data and contains daily weather records for Chittagong ( also other regions ), including:

Temperature (°C)
Rainfall (mm)
Humidity (%)
Sunshine (hours)
Date (year, month, day)
Dataset Access: Download from Mendeley Data

Key Features & Methodology

Machine Learning Pipeline

Data Preprocessing: Handles missing values, converts dates, and applies feature scaling.
Class Imbalance Handling: Uses SMOTE for balanced class representation.
Hyperparameter Tuning: Optimized via RandomizedSearchCV and StratifiedKFold.
Ensemble Learning: StackingClassifier combines multiple base models for enhanced performance.
Early Stopping: Implemented for XGBoost to prevent overfitting.
Flask Web Application

Flask-based API: Processes user input and returns predictions.
Bootstrap UI: A structured and responsive interface for efficient usage.
Real-Time Prediction: Displays predicted weather conditions with confidence scores.

Installation

Step 1: Clone the Repository

git clone https://github.com/your-username/weather_prediction_chittagong.git 
cd weather_prediction_chittagong

Step 2: Create and Activate a Virtual Environment
python -m venv venv source venv/bin/activate

Step 3: Install Dependencies

Install manually:

pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost flask joblib

tep 4: Add Dataset
Download Chittagong.csv and place it in the project directory.

Usage

Training the Model Run the following command to preprocess data, train models, and build the ensemble classifier:

python train_model.py

What Happens?

Loads and preprocesses Chittagong.csv
Applies SMOTE to balance classes
Tunes Random Forest, Gradient Boosting, XGBoost, SVM, and Logistic Regression
Builds a Stacking Ensemble
Saves the best model (stacking_ensemble.joblib) and scaler (scaler.joblib)
Running the Flask App
Start the web application with:
python app.py Then, open a browser and visit:
http://127.0.0.1:5000

Using the Web Interface

Enter weather parameters: Rainfall, Temperature, Humidity, Sunshine, and Date
Click "Predict Weather"
View predictions: Displays the predicted weather condition with confidence scores
Results The stacking ensemble achieves higher accuracy and better generalization compared to individual models. The implementation prevents overfitting using cross-validation and early stopping. The performance of models is evaluated through:

Accuracy scores
Confusion matrix
Classification report
Model comparison charts

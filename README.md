<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Weather Prediction for Chittagong</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 40px;
            padding: 20px;
            background-color: #f4f4f4;
        }
        h1 {
            font-size: 32px;
            color: #333;
        }
        h2 {
            font-size: 26px;
            color: #555;
        }
        h3 {
            font-size: 22px;
            color: #777;
        }
        p {
            font-size: 18px;
            color: #444;
        }
        code {
            background: #eee;
            padding: 5px;
            border-radius: 5px;
            font-size: 16px;
        }
        pre {
            background: #333;
            color: #fff;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }
        a {
            color: #007BFF;
            font-weight: bold;
        }
        ul {
            font-size: 18px;
        }
        strong {
            font-size: 20px;
        }
    </style>
</head>
<body>

    <h1>🌤️ Weather Prediction for Chittagong</h1>
    <p>A <strong>machine learning-based weather prediction system</strong> designed to forecast weather conditions in Chittagong, Bangladesh, using historical meteorological data. The project applies <strong>advanced data preprocessing, hyperparameter tuning, and ensemble learning</strong> to enhance prediction accuracy and generalization.</p>
    <p>A <strong>Flask web application</strong> with a structured Bootstrap UI allows users to input weather parameters and obtain real-time predictions.</p>

    <h2>📌 Introduction</h2>
    <p>Accurate weather forecasting is essential for agriculture, transportation, disaster management, and daily planning. This project builds a predictive model trained on historical weather data, integrating <strong>RandomizedSearchCV, SMOTE</strong>, and a <strong>stacking ensemble</strong> of multiple classifiers to improve accuracy and reduce overfitting.</p>
    <p>The web interface provides a structured user experience and enables real-time predictions with probability scores for different weather conditions.</p>

    <h2>📊 Dataset</h2>
    <p>The dataset is obtained from <strong>Mendeley Data</strong> and contains daily weather records for Chittagong (also other regions), including:</p>
    <ul>
        <li><strong>Temperature (°C)</strong></li>
        <li><strong>Rainfall (mm)</strong></li>
        <li><strong>Humidity (%)</strong></li>
        <li><strong>Sunshine (hours)</strong></li>
        <li><strong>Date (year, month, day)</strong></li>
    </ul>
    <p><strong>Dataset Access:</strong> <a href="https://data.mendeley.com/datasets/tbrhznpwg9/1" target="_blank">Download from Mendeley Data</a></p>

    <h2>🔧 Key Features & Methodology</h2>

    <h3>📌 Machine Learning Pipeline</h3>
    <ul>
        <li><strong>Data Preprocessing:</strong> Handles missing values, converts dates, and applies feature scaling.</li>
        <li><strong>Class Imbalance Handling:</strong> Uses SMOTE for balanced class representation.</li>
        <li><strong>Hyperparameter Tuning:</strong> Optimized via RandomizedSearchCV and StratifiedKFold.</li>
        <li><strong>Ensemble Learning:</strong> StackingClassifier combines multiple base models for enhanced performance.</li>
        <li><strong>Early Stopping:</strong> Implemented for XGBoost to prevent overfitting.</li>
    </ul>

    <h3>🌐 Flask Web Application</h3>
    <ul>
        <li><strong>Flask-based API:</strong> Processes user input and returns predictions.</li>
        <li><strong>Bootstrap UI:</strong> A structured and responsive interface for efficient usage.</li>
        <li><strong>Real-Time Prediction:</strong> Displays predicted weather conditions with confidence scores.</li>
    </ul>

    <h2>⚙️ Installation</h2>

    <h3>Step 1: Clone the Repository</h3>
    <pre>git clone https://github.com/your-username/weather_prediction_chittagong.git
cd weather_prediction_chittagong</pre>

    <h3>Step 2: Create and Activate a Virtual Environment</h3>
    <pre>python -m venv venv
source venv/bin/activate</pre>

    <h3>Step 3: Install Dependencies</h3>
    <p>Install manually:</p>
    <pre>pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost flask joblib</pre>

    <h3>Step 4: Add Dataset</h3>
    <p>Download <code>Chittagong.csv</code> and place it in the project directory.</p>

    <h2>🚀 Usage</h2>

    <h3>📌 Training the Model</h3>
    <p>Run the following command to preprocess data, train models, and build the ensemble classifier:</p>
    <pre>python train_model.py</pre>

    <h3>What Happens?</h3>
    <ul>
        <li>Loads and preprocesses <code>Chittagong.csv</code></li>
        <li>Applies SMOTE to balance classes</li>
        <li>Tunes Random Forest, Gradient Boosting, XGBoost, SVM, and Logistic Regression</li>
        <li>Builds a Stacking Ensemble</li>
        <li>Saves the best model (<code>stacking_ensemble.joblib</code>) and scaler (<code>scaler.joblib</code>)</li>
    </ul>

    <h3>🌍 Running the Flask App</h3>
    <p>Start the web application with:</p>
    <pre>python app.py</pre>
    <p>Then, open a browser and visit:</p>
    <a href="http://127.0.0.1:5000" target="_blank">http://127.0.0.1:5000</a>

    <h3>🖥️ Using the Web Interface</h3>
    <ul>
        <li>Enter weather parameters: <strong>Rainfall, Temperature, Humidity, Sunshine, and Date</strong></li>
        <li>Click <strong>"Predict Weather"</strong></li>
        <li>View predictions: Displays the predicted weather condition with confidence scores</li>
    </ul>

    <h2>📈 Results</h2>
    <p>The stacking ensemble achieves <strong>higher accuracy</strong> and better generalization compared to individual models. The implementation prevents overfitting using cross-validation and early stopping.</p>
    <p>The performance of models is evaluated through:</p>
    <ul>
        <li><strong>Accuracy scores</strong></li>
        <li><strong>Confusion matrix</strong></li>
        <li><strong>Classification report</strong></li>
        <li><strong>Model comparison charts</strong></li>
    </ul>

</body>
</html>

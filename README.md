# **Weather Prediction for Chittagong**  

A machine learning-based **weather prediction system** designed to forecast weather conditions in **Chittagong, Bangladesh** using historical meteorological data. The project applies **advanced data preprocessing, hyperparameter tuning, and ensemble learning** to enhance prediction accuracy and generalization. A **Flask web application** with a structured **Bootstrap UI** allows users to input weather parameters and obtain real-time predictions.

---

## **Table of Contents**  

1. [Introduction](#introduction)  
2. [Dataset](#dataset)  
3. [Methodology](#methodology)  
4. [Installation](#installation)  
5. [Usage](#usage)  
   - [Training the Model](#training-the-model)  
   - [Running the Flask Application](#running-the-flask-application)  
   - [Using the Web Interface](#using-the-web-interface)  
6. [Results](#results)  
7. [Future Enhancements](#future-enhancements)  



## **Introduction**  

Weather forecasting is a crucial component in various sectors, including agriculture, transportation, and disaster management. This project leverages **historical weather data** to develop a predictive model capable of classifying weather conditions into **Rain, High Temperature, Normal Temperature, or Cold**.  

By integrating **RandomizedSearchCV**, **SMOTE**, and a **stacking ensemble**, the model achieves **higher accuracy** while minimizing the risks of overfitting. A **Flask-based web application** provides a structured interface where users can input relevant weather parameters and obtain real-time predictions with confidence scores.

---

## **Dataset**  

The dataset used for this project is obtained from **Mendeley Data**, consisting of **daily meteorological records** for Chittagong. It includes the following attributes:

- **Temperature (°C)**  
- **Rainfall (mm)**  
- **Humidity (%)**  
- **Sunshine (hours)**  
- **Date (year, month, day)**  

### **Accessing the Dataset**  
The dataset can be downloaded from:  
[**Mendeley Data - Chittagong Weather**](https://data.mendeley.com/datasets/tbrhznpwg9/1)  

After downloading, rename the file as **`Chittagong.csv`** and place it in the project directory.


## **Methodology**  

### **Data Preprocessing**  

- **Handling Missing Values:**  
  - Median imputation for numerical features.  

- **Feature Engineering:**  
  - Converting **Year, Month, and Day** into a **single Date feature** in ordinal format.  
  - Categorizing weather conditions into four distinct classes.  

- **Class Imbalance Handling:**  
  - **SMOTE (Synthetic Minority Over-sampling Technique)** ensures balanced class representation in the training set.  

### **Model Development & Hyperparameter Tuning**  

- **Algorithms Implemented:**  
  - **Random Forest**  
  - **Gradient Boosting**  
  - **XGBoost (Extreme Gradient Boosting)**  
  - **SVM (Support Vector Machine)**  
  - **Logistic Regression**  

- **Optimization Strategy:**  
  - **RandomizedSearchCV** with **Stratified K-Fold cross-validation**.  
  - **Early stopping** applied to **XGBoost** for preventing overfitting.  

### **Stacking Ensemble**  

- A **StackingClassifier** is implemented to **combine multiple models** and improve predictive performance.  
- **Meta-learner:** **Logistic Regression**, trained on the predictions of the base classifiers.  

### **Flask Web Application**  

- **REST API** built using Flask.  
- **Bootstrap-based UI** for user input and predictions.  
- **Scalable and modular architecture** ensuring ease of deployment and future improvements.  


## **Installation**  

### **Step 1: Clone the Repository**  
```bash
git clone https://github.com/your-username/weather_prediction_chittagong.git
cd weather_prediction_chittagong
```

### **Step 2: Create and Activate a Virtual Environment**  
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### **Step 3: Install Dependencies**  
```bash
pip install -r requirements.txt
```

If `requirements.txt` is unavailable, install manually:  
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost flask joblib
```

### **Step 4: Add Dataset**  
Download `Chittagong.csv` from **Mendeley Data** and place it in the project directory.

---

## **Usage**  

### **Training the Model**  

To **train the model** and build the ensemble classifier, execute:  
```bash
python train_model.py
```

#### **What Happens?**  
- Loads and preprocesses `Chittagong.csv`.  
- Applies **SMOTE** to balance class distributions.  
- **Tunes hyperparameters** for Random Forest, Gradient Boosting, XGBoost, SVM, and Logistic Regression.  
- Constructs a **stacking ensemble** using optimized models.  
- Saves the trained model as **`stacking_ensemble.joblib`** and the scaler as **`scaler.joblib`**.  


### **Running the Flask Application**  

To launch the **Flask web application**, execute:  
```bash
python app.py
```
By default, the application will run on:  
[**http://127.0.0.1:5000**](http://127.0.0.1:5000)

---

### **Using the Web Interface**  

1. **Enter weather parameters**:  
   - Rainfall (mm)  
   - Temperature (°C)  
   - Humidity (%)  
   - Sunshine (hours)  
   - Date (YYYY-MM-DD format)  

2. **Click "Predict Weather"**  

3. **View Predictions**:  
   - The **predicted weather condition** is displayed.  
   - **Confidence scores** for all weather classes are provided.  


## **Results**  

The **stacking ensemble** consistently outperforms individual models by **leveraging multiple classifiers** and optimizing generalization.  

- **Higher Accuracy:**  
  The ensemble achieves superior **classification accuracy** compared to standalone models.  
- **Reduced Overfitting:**  
  Early stopping and cross-validation strategies **prevent model overfitting**.  
- **Performance Evaluation Metrics:**  
  - **Confusion Matrix**  
  - **Precision, Recall, and F1-Score**  
  - **Model Comparison Charts**  

During training, performance metrics and visualizations are **displayed and logged** for further analysis.


## **Future Enhancements**  

- **Integration of Deep Learning Models**  
  - Implement **LSTM-based Recurrent Neural Networks** for sequential weather prediction.  

- **Real-Time Weather API Integration**  
  - Incorporate live weather data for **real-time forecasts and comparisons**.  

- **Automated Model Retraining**  
  - Develop **continuous learning pipelines** for periodic retraining with updated datasets.  

# 🕵️‍♂️ Credit Card Fraud Detection
This project implements a machine learning pipeline to detect fraudulent credit card transactions using classification models. It uses a real-world inspired dataset and evaluates multiple algorithms to identify the best performer.

## 📂 Project Structure
```bash
fraud-detection/
├── fraud_detection.py      
├── requirements.txt        
├── README.md
```

## 🧠 Models Used
This script compares the performance of 3 classifiers: Logistic Regression Decision Tree Random Forest Each model is trained to classify whether a transaction is fraudulent (is_fraud = 1) or legitimate (is_fraud = 0).

## 🛠 Features of the Script
Reads and cleans transaction data Encodes categorical features Standardizes numerical features Trains models using scikit-learn Evaluates performance using: Classification Report (Precision, Recall, F1-score) AUC Score Confusion Matrix Heatmap

## 📊 Sample Output
```bash
Training Random Forest...

Random Forest Classification Report:
              precision    recall  f1-score   support

           0       0.99      1.00      0.99     29000
           1       0.92      0.85      0.88       500

    accuracy                           0.99     29500
   macro avg       0.95      0.92      0.94     29500
weighted avg       0.99      0.99      0.99     29500

Random Forest AUC Score: 0.9754
```
A heatmap of the confusion matrix is also displayed using seaborn.

## 📦 Installation
```bash
pip install -r requirements.txt
```
## 🚀 How to Run
1. Place your dataset as fraud_data.csv in the same folder.

2. Run the script:
```bash
python fraud_detection.py
```
3. The script will: Train all models

Print metrics for each

Show a confusion matrix heatmap for the Random Forest model

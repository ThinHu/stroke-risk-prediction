# Stroke Risk Prediction Pipeline

An end-to-end machine learning pipeline designed to predict stroke risk using highly imbalanced medical data. This project emphasizes robust preprocessing, advanced resampling techniques, and the comparison of complex ensemble architectures versus threshold-tuned linear models.

## Main Features

* **Automated Data Pipeline:** Seamlessly fetches the Stroke Prediction dataset directly from Kaggle using `kagglehub`.
* **Advanced Imputation:** Utilizes a Random Forest Regressor to dynamically predict and impute missing BMI values based on patient profiles, preserving data integrity better than mean/median imputation.
* **Strategic Feature Encoding:** Combines Target Encoding for complex nominal variables (like occupation and smoking status) with Label Encoding for binary features to prevent target leakage.
* **Aggressive Resampling (SMOTEENN):** Addresses severe class imbalance by synthesizing minority class examples (SMOTE-NC for categorical compatibility) and cleaning noisy boundaries (Edited Nearest Neighbours).
* **Custom Ensemble Architectures:** Implements a custom **Stacking Classifier** and **Voting Classifier** using a WEKA-matched suite of base learners: Random Forest, heavily pruned/unpruned Decision Trees (J48/RepTree equivalents), and a dynamically sized Multilayer Perceptron (MLP).
* **Medical Diagnostic Optimization (Threshold Tuning):** Includes a dedicated Logistic Regression workflow that prioritizes **Recall** (minimizing false negatives) through systematic probability threshold tuning, a critical requirement for medical screening tools.

## Project Structure

```text
stroke_prediction_project/
│
├── data/                       # Directory for raw and processed data (gitignored)
├── src/                        # Modularized source code
│   ├── __init__.py
│   ├── data_loader.py          # Kaggle dataset fetching logic
│   ├── preprocessing.py        # Cleaning, RF imputation, encoding, and SMOTEENN
│   ├── models.py               # Ensemble pipelines and WEKA-style MLP definitions
│   └── evaluation.py           # Classification metrics and visualization
│
├── main.py                     # Main execution script for the entire pipeline
├── notebook.ipynb
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```
## Acknowledgement
Special acknowledgement to the methodologies and baseline configurations discussed in the paper: "Stroke Risk Prediction with Machine Learning Techniques".

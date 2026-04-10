import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTENC
from imblearn.combine import SMOTEENN
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def clean_base_data(df):
    """Removes IDs and problematic categories."""
    df = df.drop(columns=['id'], errors='ignore')
    return df[df['gender'] != 'Other'].copy()

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

def impute_bmi_with_rf(X_train, X_test):
    """
    Uses Random Forest to predict and fill missing BMI values WITHOUT Data Leakage.
    Model is trained ONLY on X_train's non-missing data, then applied to both sets.
    """
    # Tạo bản sao để tránh cảnh báo SettingWithCopyWarning của Pandas
    X_train = X_train.copy()
    X_test = X_test.copy()

    # 1. Tách phần dữ liệu KHÔNG bị thiếu từ tập TRAIN để dạy model
    train_valid = X_train[X_train['bmi'].notna()]
    
    if train_valid.empty:
        return X_train, X_test # Safety check
        
    X_fit = train_valid.drop(columns=['bmi'])
    y_fit = train_valid['bmi']
    
    # 2. Pipeline xử lý (Giữ nguyên code cực chuẩn của bạn)
    preprocess_bmi = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['work_type', 'smoking_status', 'gender', 'ever_married', 'Residence_type']),
            ('num', StandardScaler(), ["age", "avg_glucose_level"]),
            ('pass', 'passthrough', ["hypertension", "heart_disease"])
        ]
    )
    
    model = Pipeline(steps=[
        ('preprocess', preprocess_bmi),
        ('rf', RandomForestRegressor(n_estimators=500, max_depth=10, random_state=42, n_jobs=-1))
    ])
    
    model.fit(X_fit, y_fit)
    
    missing_train = X_train[X_train['bmi'].isna()]
    if not missing_train.empty:
        X_train.loc[missing_train.index, 'bmi'] = model.predict(missing_train.drop(columns=['bmi']))
        
    missing_test = X_test[X_test['bmi'].isna()]
    if not missing_test.empty:
        X_test.loc[missing_test.index, 'bmi'] = model.predict(missing_test.drop(columns=['bmi']))
        
    return X_train, X_test

def log_transform(X):
    """Applies log transformation to specified columns."""
    X['glucose_log'] = np.log1p(X['avg_glucose_level'])
    X['bmi_log'] = np.log1p(X['bmi'])

    return X.drop(columns=['avg_glucose_level', 'bmi'], errors='ignore')

def apply_smoteenn(X_train, y_train, categorical_cols):
    """Applies SMOTENC followed by ENN for aggressive balancing."""
    cat_indices = [X_train.columns.get_loc(col) for col in categorical_cols]
    smotenc = SMOTENC(categorical_features=cat_indices, random_state=42)
    smote_enn = SMOTEENN(smote=smotenc, random_state=42)
    
    return smote_enn.fit_resample(X_train, y_train)

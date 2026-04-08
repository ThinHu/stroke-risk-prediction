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
    
    # 3. CHỈ HỌC (FIT) TRÊN TẬP TRAIN
    model.fit(X_fit, y_fit)
    
    # 4. Dự đoán và điền khuyết cho tập TRAIN
    missing_train = X_train[X_train['bmi'].isna()]
    if not missing_train.empty:
        X_train.loc[missing_train.index, 'bmi'] = model.predict(missing_train.drop(columns=['bmi']))
        
    # 5. Dự đoán và điền khuyết cho tập TEST (Sử dụng model đã học ở bước 3)
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


def apply_custom_encoding(X_train, X_test, y_train, save_dir=None):
    """
    Applies Label Encoding and Target Encoding.
    If save_dir is provided, exports the fitted encoders for web deployment.
    """
    X_train_enc = X_train.copy()
    X_test_enc = X_test.copy()
    
    # 1. Handle 'Unknown' smoking status
    X_train_enc['smoking_status'] = X_train_enc['smoking_status'].replace('Unknown', np.nan)
    X_test_enc['smoking_status'] = X_test_enc['smoking_status'].replace('Unknown', np.nan)
    
    # 2. Label Encoding
    cols_to_labl = ['gender', 'ever_married', 'Residence_type']
    label_encoders = {} # Tạo từ điển lưu trữ
    
    for col in cols_to_labl:
        le = LabelEncoder()
        X_train_enc[col] = le.fit_transform(X_train_enc[col])
        X_test_enc[col] = le.transform(X_test_enc[col])
        label_encoders[col] = le # Lưu lại encoder đã fit
        
    # 3. Target Encoding
    cols_to_target = ['work_type', 'smoking_status']
    target_encoders = {} # Tạo từ điển lưu trữ
    
    for col in cols_to_target:
        te = ce.TargetEncoder(cols=[col])
        X_train_enc[col] = te.fit_transform(X_train_enc[col], y_train)
        X_test_enc[col] = te.transform(X_test_enc[col])
        target_encoders[col] = te # Lưu lại encoder đã fit
        
    # 4. Lưu ra file nếu có yêu cầu (Dành cho Web)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(label_encoders, os.path.join(save_dir, 'label_encoders.pkl'))
        joblib.dump(target_encoders, os.path.join(save_dir, 'target_encoders.pkl'))
        print(f"Đã lưu các encoders tại: {save_dir}")
        
    return X_train_enc, X_test_enc
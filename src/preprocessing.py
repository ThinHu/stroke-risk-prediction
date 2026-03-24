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

def clean_base_data(df):
    """Removes IDs and problematic categories."""
    df = df.drop(columns=['id'], errors='ignore')
    return df[df['gender'] != 'Other'].copy()

def impute_bmi_with_rf(X):
    """Uses a Random Forest to predict and fill missing BMI values."""
    train_X = X[X['bmi'].notna()]
    missing_X = X[X['bmi'].isna()]
    
    if missing_X.empty:
        return X
        
    X_train = train_X.drop(columns=['bmi'])
    y_train = train_X['bmi']
    
    # Preprocessor for imputation
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
    
    model.fit(X_train, y_train)
    X.loc[missing_X.index, 'bmi'] = model.predict(missing_X.drop(columns=['bmi']))
    return X

def discretize_features(X):
    """Bins continuous variables into categories."""
    X['bmi_cat'] = pd.cut(X['bmi'], bins=[0, 19, 25, 30, 40, 60, 10000], labels=False)
    X['age_cat'] = pd.cut(X['age'], bins=[0, 13, 18, 45, 60, 100], labels=False)
    X['glucose_cat'] = pd.cut(X['avg_glucose_level'], bins=[0, 90, 160, 230, 500], labels=False)
    return X

def apply_smoteenn(X_train, y_train, categorical_cols):
    """Applies SMOTENC followed by ENN for aggressive balancing."""
    cat_indices = [X_train.columns.get_loc(col) for col in categorical_cols]
    smotenc = SMOTENC(categorical_features=cat_indices, random_state=42)
    smote_enn = SMOTEENN(smote=smotenc, random_state=42)
    
    return smote_enn.fit_resample(X_train, y_train)

def apply_custom_encoding(X_train, X_test, y_train):
    """
    Applies Label Encoding to simple categoricals and 
    Target Encoding to complex categoricals.
    """
    X_train_enc = X_train.copy()
    X_test_enc = X_test.copy()
    
    # 1. Handle 'Unknown' smoking status
    X_train_enc['smoking_status'] = X_train_enc['smoking_status'].replace('Unknown', np.nan)
    X_test_enc['smoking_status'] = X_test_enc['smoking_status'].replace('Unknown', np.nan)
    
    # 2. Label Encoding for binary/simple categorical columns
    cols_to_labl = ['gender', 'ever_married', 'Residence_type']
    for col in cols_to_labl:
        le = LabelEncoder()
        X_train_enc[col] = le.fit_transform(X_train_enc[col])
        X_test_enc[col] = le.transform(X_test_enc[col])
        
    # 3. Target Encoding for complex categorical columns
    cols_to_target = ['work_type', 'smoking_status']
    for col in cols_to_target:
        te = ce.TargetEncoder(cols=[col])
        X_train_enc[col] = te.fit_transform(X_train_enc[col], y_train)
        X_test_enc[col] = te.transform(X_test_enc[col])
        
    return X_train_enc, X_test_enc
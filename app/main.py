# app/main.py
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
import joblib
import os
import uvicorn

app = FastAPI(title="Hệ thống Dự đoán Đột quỵ")

# Cấu hình thư mục chứa giao diện HTML và CSS
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# 1. Tải mô hình và encoders vào bộ nhớ khi server khởi động
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "model_artifacts")

label_encoders = joblib.load(os.path.join(ARTIFACTS_DIR, "label_encoders.pkl"))
target_encoders = joblib.load(os.path.join(ARTIFACTS_DIR, "target_encoders.pkl"))
model_pipeline = joblib.load(os.path.join(ARTIFACTS_DIR, "lr_pipeline.pkl"))

# 2. Định nghĩa cấu trúc dữ liệu mới (Xóa bmi, Thêm weight và height)
class PatientData(BaseModel):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    Residence_type: str
    avg_glucose_level: float
    weight: float   # Cân nặng (kg)
    height: float   # Chiều cao (cm)
    smoking_status: str

# 3. Logic dự đoán chính
@app.post("/predict_api")
def predict_stroke(patient: PatientData):
    # Biến data từ API thành DataFrame 1 dòng
    df = pd.DataFrame([patient.dict()])
    
    # Tiền xử lý 'Unknown'
    df['smoking_status'] = df['smoking_status'].replace('Unknown', np.nan)
    
    # -------------------------------------------------------------
    # BƯỚC MỚI: TÍNH BMI VÀ RỜI RẠC HÓA DỮ LIỆU (DISCRETIZE)
    # -------------------------------------------------------------
    # 1. Tính BMI từ Cân nặng (kg) và Chiều cao (cm -> m)
    df['bmi'] = df['weight'] / ((df['height'] / 100.0) ** 2)
    
    # 2. Rời rạc hóa (Chia bins) y hệt hàm discretize_features
    df['bmi_cat'] = pd.cut(df['bmi'], bins=[0, 19, 25, 30, 40, 60, 10000], labels=False)
    df['age_cat'] = pd.cut(df['age'], bins=[0, 13, 18, 45, 60, 100], labels=False)
    df['glucose_cat'] = pd.cut(df['avg_glucose_level'], bins=[0, 90, 160, 230, 500], labels=False)
    
    # 3. Xóa cột weight và height để đưa về đúng form gốc của Pipeline
    df = df.drop(columns=['weight', 'height'])
    # -------------------------------------------------------------

    # Xử lý Label Encoding
    for col, le in label_encoders.items():
        try:
            df[col] = le.transform(df[col])
        except ValueError:
            df[col] = 0 
            
    # Xử lý Target Encoding
    for col, te in target_encoders.items():
        df[col] = te.transform(df[col])
        
    # Chạy qua Pipeline
    probability = model_pipeline.predict_proba(df)[0][1]
    
    # Áp dụng ngưỡng 0.45 
    is_stroke_risk = 1 if probability >= 0.45 else 0
    
    return {
        "risk_probability": round(probability * 100, 2),
        "is_at_risk": bool(is_stroke_risk)
    }

# 4. Các Route trả về giao diện HTML
@app.get("/")
def home_page(request: Request):
    # Sử dụng tham số có tên (keyword arguments) request và context
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/first-aid")
def first_aid_page(request: Request):
    return templates.TemplateResponse(request=request, name="first_aid.html")

@app.get("/lifestyle")
def lifestyle_page(request: Request):
    return templates.TemplateResponse(request=request, name="lifestyle.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
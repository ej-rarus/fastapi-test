from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import joblib
import pandas as pd

# FastAPI 앱 생성
app = FastAPI()

# --------------------------
# 1) 모델 로드 (앱 시작 시 1회만)
# --------------------------
bundle = joblib.load("models/model.joblib")
model = bundle["model"]
features = bundle["features"]
threshold = bundle["threshold"]

# --------------------------
# 2) 요청 스키마 정의
# --------------------------
class Features(BaseModel):
    Age: float
    BMI: float
    BloodPressure: Optional[float] = None
    Glucose: Optional[float] = None

# --------------------------
# 3) 엔드포인트들
# --------------------------
@app.get("/")
def home():
    return {"msg": "Hello, FastAPI on Mac!"}

@app.get("/ping")
def ping():
    return {"msg": "pong"}

@app.post("/predict")
def predict(data: Features):
    # dict → DataFrame (모델 피처 순서 맞추기)
    x = pd.DataFrame([data.dict()])[features]

    # 확률 예측
    prob = float(model.predict_proba(x)[:, 1][0])
    label = int(prob >= threshold)

    return {
        "inputs": data.dict(),
        "probability": prob,
        "label": label,  # 1=당뇨 위험, 0=정상
        "threshold": threshold
    }

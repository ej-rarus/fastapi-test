import joblib
import pandas as pd

# 모델 불러오기
bundle = joblib.load("models/model.joblib")
model = bundle["model"]
threshold = bundle["threshold"]

# 테스트 입력 (예시 한 건)
sample = {
    "Age": 45,
    "Gender": "Male",
    "BMI": 28.5,
    "BloodPressure": 85,
    "Insulin": 100,
    "Glucose": 130,
    "DiabetesPedigreeFunction": 0.8
}

# 예측
X = pd.DataFrame([sample])
prob = model.predict_proba(X)[:,1][0]
label = int(prob >= threshold)

print(f"예측 확률: {prob:.3f}")
print(f"분류 결과: {label} (1=당뇨 위험, 0=정상)")

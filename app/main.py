from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"msg": "Hello, FastAPI on Mac!"}

@app.get("/ping")
def ping():
    return {"msg": "pong"}

@app.post("/predict")
def predict(age: int, bmi: float, glucose: float):
    # 간단한 예시 로직
    risk = 1 if glucose > 120 else 0
    return {"age": age, "bmi": bmi, "glucose": glucose, "risk": risk}

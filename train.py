import json
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import dump

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

# ----------------------------
# 설정
# ----------------------------
DATA_PATH = Path("data/diabetes_dataset.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42
N_SPLITS = 5

# ----------------------------
# 1) 데이터 로드 & 컬럼 선택
# ----------------------------
df = pd.read_csv(DATA_PATH)

# 타깃
y = df["Outcome"].astype(int)

# ✅ 입력 변수 축소: 필수(Age, BMI) + 선택(BloodPressure, Glucose)
features = ["Age", "BMI", "BloodPressure", "Glucose"]
X = df[features].copy()

# ----------------------------
# 2) 전처리 + 모델 정의
#    - 결측 허용: SimpleImputer(median) → FastAPI에서 선택 입력 None 처리 대응
#    - 스케일링: RobustScaler
#    - 분류기: LogisticRegression(balanced) + 확률 캘리브레이션
# ----------------------------
pre = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", RobustScaler())
        ]), features),
    ],
    remainder="drop",
)

base = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
clf = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=5)

pipe = Pipeline([("pre", pre), ("clf", clf)])

# ----------------------------
# 3) 교차검증 평가 (Out-of-fold 확률)
# ----------------------------
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
probs = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]

roc_auc = roc_auc_score(y, probs)
ap = average_precision_score(y, probs)

# 임계값 튜닝 (Youden’s J = TPR - FPR 최대)
fpr, tpr, thresholds = roc_curve(y, probs)
youden = tpr - fpr
best_idx = int(np.argmax(youden))
best_thresh = float(thresholds[best_idx])

preds = (probs >= best_thresh).astype(int)
f1 = f1_score(y, preds)

print(f"[CV] ROC-AUC={roc_auc:.3f} | PR-AUC={ap:.3f} | F1@{best_thresh:.2f}={f1:.3f}")

# ----------------------------
# 4) 전체 데이터로 최종 적합 & 저장
# ----------------------------
pipe.fit(X, y)

bundle = {
    "model": pipe,
    "features": features,        # ✅ 추후 입력 검증/정렬에 사용
    "threshold": best_thresh,
    "seed": SEED,
}
dump(bundle, MODEL_DIR / "model.joblib")

meta = {
    "roc_auc": float(roc_auc),
    "avg_precision": float(ap),
    "f1_at_threshold": float(f1),
    "threshold": best_thresh,
    "n_splits": N_SPLITS,
    "n_samples": int(len(df)),
    "features": features,
}
with open(MODEL_DIR / "model_meta.json", "w") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print("[SAVE] models/model.joblib, models/model_meta.json 저장 완료")

import json, os
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import dump

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_recall_curve, roc_curve)

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV


DATA_PATH = Path("data/diabetes_dataset.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42
N_SPLITS = 5

# ----------------------------
# 1) 타깃/피처 분리
# ----------------------------
df = pd.read_csv(DATA_PATH)   # 👈 이 줄 추가


y = df["Outcome"].astype(int)
X = df.drop(columns=["Outcome", "PatientID"])  # PatientID는 제거

# 수치/범주형 구분
num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
cat_cols = [c for c in X.columns if c not in num_cols]

print("수치형:", num_cols)
print("범주형:", cat_cols)

# ----------------------------
# 2) 전처리 + 모델 정의
# ----------------------------
pre = ColumnTransformer(
    transformers=[
        ("num", RobustScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ],
    remainder="drop",
)

base = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
clf = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=5)

pipe = Pipeline([("pre", pre), ("clf", clf)])

# ----------------------------
# 3) 교차검증 평가
# ----------------------------
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
probs = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]

roc_auc = roc_auc_score(y, probs)
ap = average_precision_score(y, probs)

# 임계값 튜닝 (Youden’s J)
fpr, tpr, thresholds = roc_curve(y, probs)
youden = tpr - fpr
best_idx = youden.argmax()
best_thresh = float(thresholds[best_idx])

preds = (probs >= best_thresh).astype(int)
f1 = f1_score(y, preds)

print(f"[CV] ROC-AUC={roc_auc:.3f} | PR-AUC={ap:.3f} | F1@{best_thresh:.2f}={f1:.3f}")

# ----------------------------
# 4) 최종 학습 & 저장
# ----------------------------
pipe.fit(X, y)

bundle = {
    "model": pipe,
    "num_cols": num_cols,
    "cat_cols": cat_cols,
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
    "features": list(X.columns),
}
with open(MODEL_DIR / "model_meta.json", "w") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print("[SAVE] models/model.joblib, models/model_meta.json 저장 완료")

# --- dlp_core.py (Model-only: XGBoost decides) ---
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
import joblib

# -------- Load artifacts --------
# Make this relative if you want portability:
# ART = (Path(__file__).resolve().parents[1] / "artifacts")
ART = Path(r"C:/Users/yashf/Desktop/DLP-Project/artifacts")

vectorizer = joblib.load(ART / "tfidf_char35.joblib")
model      = joblib.load(ART / "xgb_char35.joblib")
policy     = json.loads((ART / "policy.json").read_text(encoding="utf-8"))

classes = policy["classes"]
T1 = float(policy["t1"])   # WARN threshold
T2 = float(policy["t2"])   # BLOCK threshold

# Which classes count as "sensitive" for policy scoring
SENSITIVE = {"phone", "email", "address"}

def _sensitive_score(proba: np.ndarray) -> float:
    sens_idx = [i for i, c in enumerate(classes) if c.lower() in SENSITIVE]
    return float(max((proba[i] for i in sens_idx), default=0.0))

# ---------- PUBLIC API ----------
def score_text(text: str):
    # 1) Text → TF-IDF
    X_text = vectorizer.transform([text])

    # 2) Keep feature SHAPE consistent with your trained model:
    #    You trained with 3 extra flags; feed zeros to match.
    #    If you retrain WITHOUT flags, change the next line to:
    #       X_input = X_text
    X_input = hstack([X_text, csr_matrix([[0, 0, 0]])])


    proba = model.predict_proba(X_input)[0]
    top_i = int(np.argmax(proba))
    top_cls = classes[top_i]

    # 4) Policy mapping using validation-tuned thresholds
    sens_score = _sensitive_score(proba)
    if   sens_score >= T2: action = "BLOCK"
    elif sens_score >= T1: action = "WARN"
    else:                  action = "ALLOW"

    return {
        "class":  top_cls,
        "score":  round(sens_score, 3),
        "policy": action
    }

def score_dataframe(df: pd.DataFrame):
    return [score_text(t) for t in df["text"].astype(str)]

__all__ = ["score_text", "score_dataframe"]

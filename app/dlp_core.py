# dlp_core.py — PURE XGBOOST (NO REGEX, NO RULES)

import json
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

# ---------------- Load artifacts ----------------
ART = Path(r"C:/Users/yashf/Desktop/DLP-Project/artifacts")

vectorizer = joblib.load(ART / "tfidf_char35.joblib")
model      = joblib.load(ART / "xgb_char35.joblib")
policy     = json.loads((ART / "policy.json").read_text())

CLASSES = policy["classes"]
T_WARN  = float(policy["t1"])
T_BLOCK = float(policy["t2"])

# Define which classes are considered sensitive
SENSITIVE = {"phone", "email", "cnic", "address"}

# ---------------- Core scoring ----------------
def score_text(text: str):
    X = vectorizer.transform([str(text)])
    proba = model.predict_proba(X)[0]

    # max probability among sensitive classes
    sens_score = max(
        proba[i]
        for i, c in enumerate(CLASSES)
        if c.lower() in SENSITIVE
    )

    if sens_score >= T_BLOCK:
        policy_action = "BLOCK"
    elif sens_score >= T_WARN:
        policy_action = "WARN"
    else:
        policy_action = "ALLOW"

    top_i = int(np.argmax(proba))

    return {
        "class": CLASSES[top_i],
        "score": round(float(sens_score), 3),
        "policy": policy_action
    }


def score_dataframe(df: pd.DataFrame):
    return [score_text(t) for t in df["text"].astype(str)]

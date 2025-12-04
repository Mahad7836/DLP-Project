# src/train_xgb.py
# ------------------------------------------------------------
# Train XGBoost DLP classifier (char TF-IDF + regex flags + strict phone gate)
# with safe splits, label encoding, and validation-tuned thresholds.
# Saves artifacts for the FastAPI/Streamlit demo.
# ------------------------------------------------------------
import json
import pathlib
import re

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, precision_recall_curve

from xgboost import XGBClassifier

# --------- Config ---------
ART = pathlib.Path("artifacts"); ART.mkdir(exist_ok=True)
DATA = "data/dlp_data.csv"   # expects columns: text,label
SENSITIVE = {"phone", "email", "address"}  # classes contributing to "sensitive score"


import phonenumbers
from phonenumbers import PhoneNumberType, PhoneNumberFormat

EMAIL_RE   = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
CNIC_RE    = re.compile(r'\b\d{5}-\d{7}-\d\b')   # dashed CNIC
CNIC13_RE  = re.compile(r'\b\d{13}\b')           # undashed 13-digit CNIC

ALLOWED_TYPES   = {PhoneNumberType.MOBILE}
ALLOWED_REGIONS = {"PK", "AE", "US", "GB", "DE", "IN"}
DEFAULT_REGION  = "PK"
CONTEXT_WORDS   = {"call", "phone", "tel", "mobile", "mob", "cell", "whatsapp", "wa", "contact"}

def has_email(s: str) -> int:
    return int(bool(EMAIL_RE.search(s or "")))

def has_cnic(s: str) -> int:
    s = s or ""
    return int(bool(CNIC_RE.search(s) or CNIC13_RE.search(s)))

def _near_ctx(text: str, i: int, j: int, win: int = 16) -> bool:
    if not text: return False
    lo = max(0, i - win); hi = min(len(text), j + win)
    ctx = text[lo:hi].lower()
    return any(w in ctx for w in CONTEXT_WORDS)

def _repetitive(digits: str) -> bool:
    return len(set(digits)) <= 2

def extract_valid_phones_strict(text: str):
    """Return list of E.164 phones after strict checks (multi-country + PK local)."""
    if not text: return []
    out = []
    for m in phonenumbers.PhoneNumberMatcher(text, DEFAULT_REGION):
        num = m.number
        if not (phonenumbers.is_possible_number(num) and phonenumbers.is_valid_number(num)):
            continue
        if phonenumbers.region_code_for_number(num) not in ALLOWED_REGIONS:
            continue
        if phonenumbers.number_type(num) not in ALLOWED_TYPES:
            continue
        nd = str(num.national_number)
        if not (8 <= len(nd) <= 12) or _repetitive(nd):
            continue
        raw = m.raw_string.strip()
        # Allow common PK local mobile (03XXXXXXXXX) even without '+'
        digits_only = re.sub(r"\D+", "", raw)
        pk_local_ok = (digits_only.startswith("03") and len(digits_only) == 11)
        if not raw.startswith("+") and not (pk_local_ok or _near_ctx(text, m.start, m.end)):
            continue
        out.append(phonenumbers.format_number(num, PhoneNumberFormat.E164))
    return out

def make_flags(series: pd.Series) -> csr_matrix:
    """Build sparse matrix of regex/phone flags for a text Series."""
    return csr_matrix(np.c_[
        series.apply(has_email),
        series.apply(lambda s: int(bool(extract_valid_phones_strict(s)))),
        series.apply(has_cnic)
    ])

# --------- Safe split utility ---------
def safe_split(X_series: pd.Series, y_array: np.ndarray):
    """Create 70/15/15 train/val/test with stratification when possible (handles rare classes)."""
    y_counts = pd.Series(y_array).value_counts()
    strat = y_array if (y_counts >= 2).all() else None

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_series, y_array, test_size=0.15, random_state=42, stratify=strat
    )

    y_tr_counts = pd.Series(y_tr).value_counts()
    strat2 = y_tr if (y_tr_counts >= 2).all() else None
    val_rel = 0.15 / 0.85  # ≈0.17647 to reach 70/15/15 overall

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=val_rel, random_state=42, stratify=strat2
    )
    return X_tr, X_val, X_te, y_tr, y_val, y_te

# ============================================================
#                       MAIN
# ============================================================
if __name__ == "__main__":
    # ---- Load data & encode labels ----
    df = pd.read_csv(DATA)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("DATA must contain columns: 'text' and 'label'")
    df["text"] = df["text"].astype(str)

    le = LabelEncoder()
    y_str = df["label"].astype(str)
    y = le.fit_transform(y_str)        # numeric labels 0..K-1
    class_names = list(le.classes_)    # order matches predict_proba columns

    # ---- Split ----
    X_tr_txt, X_val_txt, X_te_txt, y_tr, y_val, y_te = safe_split(df["text"], y)

    # ---- Vectorize (fit on TRAIN only) ----
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5))
    X_tr_vec = vec.fit_transform(X_tr_txt)
    X_val_vec = vec.transform(X_val_txt)
    X_te_vec  = vec.transform(X_te_txt)

    # ---- Flags & fuse ----
    X_tr = hstack([X_tr_vec, make_flags(X_tr_txt)])
    X_val = hstack([X_val_vec, make_flags(X_val_txt)])
    X_te  = hstack([X_te_vec,  make_flags(X_te_txt)])

    # ---- XGBoost model ----
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42
    )
    xgb.fit(X_tr, y_tr)

    # ---- Test report (with string labels) ----
    y_te_pred = xgb.predict(X_te)
    print("\nTEST REPORT (XGBoost):\n")
    print(classification_report(
        le.inverse_transform(y_te),
        le.inverse_transform(y_te_pred),
        digits=4
    ))

    # ---- Threshold tuning on validation (t1 warn, t2 block) ----
    classes = class_names[:]                 # ['address','email','none','phone'] (example)
    lower   = [c.lower() for c in classes]
    sens_idx = [i for i, c in enumerate(lower) if c in SENSITIVE]

    proba_val = xgb.predict_proba(X_val)     # shape (N, K), columns align with label encoder order
    sens_val  = proba_val[:, sens_idx].max(axis=1) if sens_idx else np.zeros(len(y_val))
    is_sens   = pd.Series(le.inverse_transform(y_val)).str.lower().isin(SENSITIVE).astype(int).values

    # Choose t1 to maximize F1(sensitive vs none)
    best_t1, best_f1 = 0.30, -1.0
    for t in np.linspace(0.10, 0.90, 81):
        preds = (sens_val >= t).astype(int)
        f1 = f1_score(is_sens, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t1 = f1, t

    # Choose t2 (block) near high precision region; clamp to sane bounds
    prec, rec, thr = precision_recall_curve(is_sens, sens_val)
    try:
        idx = np.where(prec[:-1] >= 0.90)[0]
        best_t2 = float(thr[idx[-1]]) if len(idx) else min(0.95, best_t1 + 0.15)
    except Exception:
        best_t2 = min(0.95, best_t1 + 0.15)

    best_t2 = min(best_t2, 0.95)         # don't require perfect certainty
    best_t2 = max(best_t2, best_t1 + 0.10)  # keep warn<block gap

    print(f"\nSelected thresholds:  WARN t1={best_t1:.2f}  |  BLOCK t2={best_t2:.2f}\n")

    # ---- Save artifacts ----
    joblib.dump(vec, ART / "tfidf_char35.joblib")
    joblib.dump(xgb, ART / "xgb_char35.joblib")
    (ART / "policy.json").write_text(json.dumps({
        "classes": classes,                         # IMPORTANT: same order as predict_proba
        "t1": float(best_t1),
        "t2": float(best_t2),
        "default_region": DEFAULT_REGION,
        "allowed_regions": sorted(list(ALLOWED_REGIONS))
    }, indent=2))

    print(f"Saved artifacts to: {ART.resolve()}")

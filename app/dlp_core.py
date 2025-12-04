import re, json
import numpy as np, pandas as pd
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
import joblib
import phonenumbers
from phonenumbers import PhoneNumberType, PhoneNumberFormat

ART = Path("artifacts")
vectorizer = joblib.load(ART/"tfidf_char35.joblib")
model      = joblib.load(ART/"xgb_char35.joblib")     # XGBoost
policy     = json.loads((ART/"policy.json").read_text())
classes    = policy["classes"]; T1=float(policy["t1"]); T2=float(policy["t2"])

# --- simple email pattern ---
EMAIL_RE = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')

# --- phone validation settings ---
ALLOWED_TYPES   = {PhoneNumberType.MOBILE}
ALLOWED_REGIONS = set(policy.get("allowed_regions", ["PK","AE","US","GB","DE","IN"]))
DEFAULT_REGION  = policy.get("default_region","PK")
CONTEXT_WORDS   = {"call","phone","tel","mobile","mob","cell","whatsapp","wa","contact"}

def has_email(s: str) -> int:
    return int(bool(EMAIL_RE.search(s or "")))

def _near_ctx(text,i,j,win=16):
    if not text: return False
    lo=max(0,i-win); hi=min(len(text), j+win)
    ctx=text[lo:hi].lower()
    return any(w in ctx for w in CONTEXT_WORDS)

def _repetitive(d): return len(set(d))<=2

def extract_valid_phones_strict(text: str):
    """Strict multi-country validation, includes PK-local acceptance (03XXXXXXXXX)."""
    if not text: return []
    out=[]
    for m in phonenumbers.PhoneNumberMatcher(text, DEFAULT_REGION):
        num=m.number
        if not (phonenumbers.is_possible_number(num) and phonenumbers.is_valid_number(num)): 
            continue
        if phonenumbers.region_code_for_number(num) not in ALLOWED_REGIONS: 
            continue
        if phonenumbers.number_type(num) not in ALLOWED_TYPES: 
            continue
        nd=str(num.national_number)
        if not (8<=len(nd)<=12) or _repetitive(nd): 
            continue
        raw=m.raw_string.strip()
        digits=re.sub(r"\D+","",raw)
        pk_local = (digits.startswith("03") and len(digits)==11)
        if not raw.startswith("+") and not (pk_local or _near_ctx(text,m.start,m.end)):
            continue
        out.append(phonenumbers.format_number(num, PhoneNumberFormat.E164))
    return out

def _canon(s): return re.sub(r"\s+","",str(s or "").lower())
def _is_phone_label(name): return "phone" in _canon(name)

# We only care about these as "sensitive"
SENSITIVE = {"phone","email","address"}
cls_lower = [c.lower() for c in classes]
sens_idx  = [i for i,c in enumerate(cls_lower) if c in SENSITIVE]

def score_text(text: str):
    # 1) strict phone gate
    phones = extract_valid_phones_strict(text)
    email_flag = has_email(text)
    phone_flag = int(bool(phones))

    # Build model input:
    # NOTE: Model was trained with 3 extra flags [email_flag, phone_flag, cnic_flag].
    # CNIC is removed → pass 0 as placeholder to keep the feature count aligned.
    X_text  = vectorizer.transform([text])
    X_input = hstack([X_text, csr_matrix([[email_flag, phone_flag, 0]])])

    # If strict phone fired → force PHONE decision (bypass model uncertainty)
    if phones:
        action = "BLOCK" if 1.0 >= T2 else ("WARN" if 1.0 >= T1 else "ALLOW")
        return {
            "class":"phone",
            "score":1.0,
            "policy":action,
            "phones":phones,
            "email_flag":email_flag,
            "phone_gate":1
        }

    # 2) model probabilities (suppress 'phone' when strict gate didn't fire)
    proba = model.predict_proba(X_input)[0]
    for i,c in enumerate(classes):
        if _is_phone_label(c): 
            proba[i]=0.0

    sens_score = float(max(proba[i] for i in sens_idx)) if sens_idx else 0.0
    top_i = int(np.argmax(proba)); top_cls = classes[top_i]

    if   sens_score >= T2: action="BLOCK"
    elif sens_score >= T1: action="WARN"
    else:                  action="ALLOW"

    return {
        "class": top_cls,
        "score": round(sens_score,3),
        "policy": action,
        "phones": phones,
        "email_flag": email_flag,
        "phone_gate": 0
    }

def score_dataframe(df: pd.DataFrame):
    out=[]
    for t in df["text"].astype(str):
        out.append(score_text(t))
    return out

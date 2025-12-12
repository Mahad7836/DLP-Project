# file: api.py
import os
import re
import joblib
import unicodedata
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import phonenumbers
from phonenumbers import PhoneNumberType, PhoneNumberFormat

# ----------------------------
# 1. Settings & Artifacts
# ----------------------------
ARTIFACT_DIR = "artifacts"

try:
    print("Loading models...")
    vectorizer = joblib.load(os.path.join(ARTIFACT_DIR, "tfidf_vectorizer_xgb.joblib"))
    model = joblib.load(os.path.join(ARTIFACT_DIR, "xgboost_classifier.joblib"))
    label_encoder = joblib.load(os.path.join(ARTIFACT_DIR, "label_encoder.joblib"))
    classes = label_encoder.classes_
    print("Models loaded successfully.")
except FileNotFoundError:
    raise RuntimeError(f"Artifacts not found in {ARTIFACT_DIR}. Run training first.")

# ----------------------------
# 2. Logic Helpers
# ----------------------------
ALLOWED_REGIONS = {"PK", "AE", "US", "GB", "DE", "IN"}
ALLOWED_TYPES = {PhoneNumberType.MOBILE}
DEFAULT_REGION = "PK"
CONTEXT_WORDS = {"call", "phone", "tel", "mobile", "mob", "cell", "whatsapp", "wa", "contact"}

def normalize_text(s):
    s = unicodedata.normalize('NFKC', str(s))
    return s.strip()

def _near_context(text, i, j, win=16):
    lo = max(0, i - win)
    hi = min(len(text), j + win)
    ctx = text[lo:hi].lower()
    return any(w in ctx for w in CONTEXT_WORDS)

def _repetitive(ndigits):
    return len(set(ndigits)) <= 2

def extract_valid_phones_strict(text):
    out = []
    try:
        for m in phonenumbers.PhoneNumberMatcher(text, DEFAULT_REGION):
            num = m.number
            if not (phonenumbers.is_possible_number(num) and phonenumbers.is_valid_number(num)): continue
            region = phonenumbers.region_code_for_number(num)
            if region and region not in ALLOWED_REGIONS: continue
            if phonenumbers.number_type(num) not in ALLOWED_TYPES: continue
            nd = str(num.national_number)
            if not (8 <= len(nd) <= 12) or len(nd) == 6 or _repetitive(nd): continue
            raw = m.raw_string.strip()
            if not raw.startswith('+') and not _near_context(text, m.start, m.end): continue
            out.append(phonenumbers.format_number(num, PhoneNumberFormat.E164))
    except Exception: pass
    return out

def has_phone_strict(text):
    return int(bool(extract_valid_phones_strict(text)))

# Regex Logic
EMAIL_RE = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
ADDRESS_PATTERNS = [
    r'\b(?:street|st\.?|road|rd\.?|avenue|ave\.?|lane|ln\.?|boulevard|blvd\.?|drive|dr\.?\b|sector|phase|block|house|flat|apt|apartment)\b',
    r'\b(?:po box|p\.?o\.?\s*box)\s*\d+\b',
    r'\b\d{1,5}\s+(?:[A-Za-z0-9]+\s){0,3}(?:street|st|road|rd|avenue|ave|lane|ln)\b',
    r'\bpostal\s*code\b',
    r'\bzip\s*code\b'
]
ADDRESS_RE = re.compile('|'.join(ADDRESS_PATTERNS), flags=re.IGNORECASE)

def has_email(text): return int(bool(EMAIL_RE.search(text or "")))
def has_address(text): return int(bool(ADDRESS_RE.search(text or "")))

# ----------------------------
# 3. API Server
# ----------------------------
app = FastAPI(title="DLP AI Scanner")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health(): return {"status": "online", "model": "XGBoost v1.0"}

@app.post("/scan")
async def scan_text(payload: dict):
    text = payload.get("text", "")
    if not text: return {"error": "Empty text"}
    
    # Preprocess
    text_norm = normalize_text(text)
    text_lower = text_norm.lower()
    
    # Flags
    email_f = has_email(text_lower)
    extracted_phones = extract_valid_phones_strict(text_lower) # Get actual list
    phone_f = 1 if extracted_phones else 0
    addr_f = has_address(text_lower)
    
    # Vectorize
    vec = vectorizer.transform([text_lower])
    features = hstack([vec, csr_matrix([[email_f, phone_f, addr_f]])])
    
    # Predict
    proba = model.predict_proba(features)[0]

    # --- SANITY CHECK (The Fix for Gibberish) ---
    # If Regex didn't find a strict phone number, FORCE the model to stop thinking it's a Phone.
    if not extracted_phones:
        phone_indices = [i for i, c in enumerate(classes) if 'phone' in str(c).lower()]
        for i in phone_indices:
            proba[i] = 0.0

    # Re-calculate Top Class after removing False Phones
    top_i = int(np.argmax(proba))
    confidence = float(proba[top_i])
    pred_label = classes[top_i]

    # --- AGGRESSIVE POLICY LOGIC ---
    policy = "ALLOW"

    # 1. GIBBERISH FILTER: If confidence is garbage (< 35%), assume Safe/None
    if confidence < 0.35 and not (email_f or extracted_phones or addr_f):
        pred_label = "NONE"
        policy = "ALLOW"
        confidence = 0.0

    # 2. REGEX OVERRIDES (Block immediately)
    elif email_f == 1:
        policy = "BLOCK"
        pred_label = "EMAIL"
        confidence = max(confidence, 0.99)
    
    elif extracted_phones:
        policy = "BLOCK"
        pred_label = "PHONE"
        confidence = max(confidence, 0.99)

    # 3. MODEL PREDICTION (Block if confident)
    elif pred_label in ["PHONE", "EMAIL", "ADDRESS"] and confidence > 0.45:
        policy = "BLOCK"

    # 4. WARNING (If unsure)
    elif pred_label != "NONE" and confidence > 0.25:
        policy = "WARN"

    # Debug print
    print(f"Text: {text[:20]}... | Label: {pred_label} | Conf: {confidence:.2f} | Policy: {policy}")

    return {
        "text_preview": text[:50] + "...",
        "class": pred_label,
        "confidence": round(confidence, 2),
        "policy": policy,
        "flags": {"email": email_f, "phone": phone_f, "address": addr_f}
    }

@app.get("/shap-plot/{label}")
async def get_shap_plot(label: str):
    filename = f"shap_summary_{label.upper()}.png"
    file_path = os.path.join(ARTIFACT_DIR, filename)
    if os.path.exists(file_path): return FileResponse(file_path)
    return {"error": "Plot not found."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
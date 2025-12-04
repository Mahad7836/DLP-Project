import re
import pandas as pd
import numpy as np
import unicodedata
import os
import joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import phonenumbers
from phonenumbers import PhoneNumberType, PhoneNumberFormat

# ----------------------------
# Settings
# ----------------------------

RND = 42
CONF_THRESHOLD = 0.65
ALLOWED_REGIONS = {"PK", "AE", "US", "GB", "DE", "IN"}
ALLOWED_TYPES = {PhoneNumberType.MOBILE}
DEFAULT_REGION = "PK"
CONTEXT_WORDS = {"call", "phone", "tel", "mobile", "mob", "cell", "whatsapp", "wa", "contact"}

# ----------------------------
# Load dataset
# ----------------------------

df = pd.read_csv(r"data\newdata.csv")
df = df.dropna(subset=['text', 'label'])

def normalize_text(s):
    s = unicodedata.normalize('NFKC', str(s))
    return s.strip()

df['text_norm'] = df['text'].apply(normalize_text)
df['text_lower'] = df['text_norm'].str.lower()

# ----------------------------
# Label Encoding (Required for XGBoost)
# ----------------------------
label_encoder = LabelEncoder()
# We encode the string labels into numbers (0, 1, 2...)
df['label_num'] = label_encoder.fit_transform(df['label'])

X_text = df['text_lower']
y = df['label_num']

X_tr_text, X_te_text, y_tr, y_te = train_test_split(
    X_text, y, test_size=0.3, stratify=y, random_state=RND
)

# ----------------------------
# Regex-based feature functions
# ----------------------------

EMAIL_RE = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
CNIC_RE = re.compile(r'\b\d{5}-\d{7}-\d\b')

def has_email(text):
    return int(bool(EMAIL_RE.search(text or "")))

def has_cnic(text):
    return int(bool(CNIC_RE.search(text or "")))

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
            if not (phonenumbers.is_possible_number(num) and phonenumbers.is_valid_number(num)):
                continue
            if phonenumbers.region_code_for_number(num) not in ALLOWED_REGIONS:
                continue
            if phonenumbers.number_type(num) not in ALLOWED_TYPES:
                continue
            nd = str(num.national_number)
            if not (8 <= len(nd) <= 12) or len(nd) == 6 or _repetitive(nd):
                continue
            raw = m.raw_string.strip()
            if not raw.startswith('+') and not _near_context(text, m.start, m.end):
                continue
            out.append(phonenumbers.format_number(num, PhoneNumberFormat.E164))
    except Exception:
        pass
    return out

def has_phone_strict(text):
    return int(bool(extract_valid_phones_strict(text)))

df['email_flag'] = df['text_lower'].apply(has_email)
df['cnic_flag'] = df['text_lower'].apply(has_cnic)
df['phone_flag'] = df['text_lower'].apply(has_phone_strict)

# ----------------------------
# TF-IDF + feature stack
# ----------------------------

vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=25000)
X_tr_vec = vectorizer.fit_transform(X_tr_text)
X_te_vec = vectorizer.transform(X_te_text)

regex_tr = csr_matrix(df.loc[X_tr_text.index, ['email_flag', 'phone_flag', 'cnic_flag']].values)
regex_te = csr_matrix(df.loc[X_te_text.index, ['email_flag', 'phone_flag', 'cnic_flag']].values)

X_tr = hstack([X_tr_vec, regex_tr])
X_te = hstack([X_te_vec, regex_te])

# ----------------------------
# Train XGBoost
# ----------------------------

model = xgb.XGBClassifier(
    n_estimators=600,
    max_depth=10,        # XGB usually needs a depth limit
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob", # Returns probabilities
    random_state=RND,
    n_jobs=-1,
    tree_method="hist"   # Faster training
)

print("Training XGBoost...")
model.fit(X_tr, y_tr)

# Eval
print("Predicting on test set...")
preds_num = model.predict(X_te)

# Decode predictions back to strings for reporting
preds_str = label_encoder.inverse_transform(preds_num)
y_te_str = label_encoder.inverse_transform(y_te)

print("\n=== XGBOOST PERFORMANCE ===")
print("Accuracy:", accuracy_score(y_te_str, preds_str))
print("\nClassification Report:\n", classification_report(y_te_str, preds_str, digits=4))
print("\nConfusion Matrix:\n", confusion_matrix(y_te_str, preds_str))

# Save artifacts
os.makedirs("artifacts", exist_ok=True)
joblib.dump(vectorizer,    "artifacts/tfidf_vectorizer_xgb.joblib")
joblib.dump(model,         "artifacts/xgboost_classifier.joblib")
joblib.dump(label_encoder, "artifacts/label_encoder.joblib")

print("Artifacts saved.")

# ----------------------------
# Interactive PII Detection
# ----------------------------

def _canon(s):
    return re.sub(r'\s+', '', str(s or '').lower())

def _is_phone_label(name):
    return bool(re.search(r'phone', _canon(name)))

def _is_none_label(name):
    return _canon(name) in {'none', 'no_pii', 'nopii', 'neutral', 'other', 'negative'}

# Get class names from the encoder
classes = label_encoder.classes_

# Find the numeric indices for 'phone' and 'none' classes
phone_idx = [i for i, c in enumerate(classes) if _is_phone_label(c)]
none_idx = [i for i, c in enumerate(classes) if _is_none_label(c)]

print("\n--- Interactive PII Detection ---")
print("Type any text (email, phone, CNIC, etc.). Type 'exit' to quit.\n") 

while True:
    s = input("Enter text to check: ")
    if s.lower().strip() in ["exit", "quit"]:
        print("Exiting DLP Detector. Goodbye!")
        break

    if not s.strip():
        continue

    # Regex + phone signals
    email_flag = has_email(s)
    phone_flag = has_phone_strict(s)
    cnic_flag = has_cnic(s)

    # Transform input
    X_text_input = vectorizer.transform([s])
    
    # Check for empty vector (Unknown words check)
    if X_text_input.nnz == 0 and not (email_flag or phone_flag or cnic_flag):
        # If no known words AND no regex flags, safe to assume it's just noise/unknown
        # However, to be safe, we usually just let it predict or default to None.
        pass 

    X_input = hstack([X_text_input, csr_matrix([[email_flag, phone_flag, cnic_flag]])])

    # If strict phone exists -> force PHONE
    phones = extract_valid_phones_strict(s)
    if phones:
        print("Detected Possible PII: PHONE")
        print("Phones (E.164):", ", ".join(phones))
        print("-" * 60)
        continue

    # Otherwise, predict with XGBoost
    proba = model.predict_proba(X_input)[0]
    
    # Hard-gate phone: If regex didn't find a phone (we are here), 
    # but model thinks it's a phone, we suppress the model's phone guess.
    for i in phone_idx:
        proba[i] = 0.0 

    top_i = int(np.argmax(proba))
    confidence = proba[top_i]
    
    # Get the string label
    pred_label = classes[top_i]

    # Apply Threshold
    final = pred_label if confidence >= CONF_THRESHOLD else 'none'

    if _is_none_label(final):
        print("No sensitive information detected.")
    else:
        print(f"Detected Possible PII: {str(final).upper()}")
        print(f"Confidence: {confidence:.2f}")
    
    print("-" * 60)

import re
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import phonenumbers
from phonenumbers import PhoneNumberType, PhoneNumberFormat

# ----------------------------
# 1) Load dataset
# ----------------------------
df = pd.read_csv('dlp_dataset.csv')  # Columns: 'text', 'label'
df['text'] = df['text'].astype(str)
y = df['label'].astype(str)

# ----------------------------
# 2) Regex-based feature functions
# ----------------------------
EMAIL_RE = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
CNIC_RE  = re.compile(r'\b\d{5}-\d{7}-\d\b')

def has_email(text): return int(bool(EMAIL_RE.search(text or "")))
def has_cnic(text): return int(bool(CNIC_RE.search(text or "")))

# Global strict phone parsing
ALLOWED_REGIONS = {"PK", "AE", "US", "GB", "DE", "IN"}
ALLOWED_TYPES = {PhoneNumberType.MOBILE}
DEFAULT_REGION = "PK"
CONTEXT_WORDS = {"call","phone","tel","mobile","mob","cell","whatsapp","wa","contact"}

def _near_context(text, i, j, win=16):
    lo = max(0, i-win); hi = min(len(text), j+win)
    ctx = text[lo:hi].lower()
    return any(w in ctx for w in CONTEXT_WORDS)

def _repetitive(ndigits): return len(set(ndigits)) <= 2

def extract_valid_phones_strict(text):
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
        if not (8 <= len(nd) <= 12) or len(nd)==6 or _repetitive(nd):
            continue
        raw = m.raw_string.strip()
        if not raw.startswith('+') and not _near_context(text, m.start, m.end):
            continue
        out.append(phonenumbers.format_number(num, PhoneNumberFormat.E164))
    return out

def has_phone_strict(text): return int(bool(extract_valid_phones_strict(text)))

# ----------------------------
# 3) Add regex features
# ----------------------------
df['email_flag'] = df['text'].apply(has_email)
df['cnic_flag']  = df['text'].apply(has_cnic)
df['phone_flag'] = df['text'].apply(has_phone_strict)

# ----------------------------
# 4) TF-IDF + feature stack
# ----------------------------
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
X_vec = vectorizer.fit_transform(df['text'])
regex_sparse = csr_matrix(df[['email_flag','phone_flag','cnic_flag']].values)
X = hstack([X_vec, regex_sparse])

# ----------------------------
# 5) Train/test split
# ----------------------------
# Stratify only if all classes have >=2 samples
if df['label'].value_counts().min() >= 2:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
else:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_tr, y_tr)

# ----------------------------
# 6) Eval
# ----------------------------
print("Training Complete!\n")
print(classification_report(y_te, model.predict(X_te), digits=4))

# ----------------------------
# 7) Interactive testing
# ----------------------------
def _canon(s): return re.sub(r'\s+', '', str(s or '').lower())
def _is_phone_label(name): return bool(re.search(r'phone', _canon(name)))
def _is_none_label(name): return _canon(name) in {'none','no_pii','nopii','neutral','other','negative'}

classes = list(model.classes_)
phone_idx = [i for i,c in enumerate(classes) if _is_phone_label(c)]
none_idx  = [i for i,c in enumerate(classes) if _is_none_label(c)]
CONF_THRESHOLD = 0.65

print("\n--- Interactive PII Detection ---")
print("Type any text (email, phone, CNIC, etc.). Type 'exit' to quit.\n")

while True:
    s = input("Enter text to check: ")
    if s.lower().strip() == "exit":
        print("Exiting DLP Detector. Goodbye!")
        break

    # Strict regex signals
    email_flag = has_email(s)
    phone_flag = has_phone_strict(s)
    cnic_flag  = has_cnic(s)

    X_text = vectorizer.transform([s])
    X_input = hstack([X_text, csr_matrix([[email_flag, phone_flag, cnic_flag]])])

    # If strict phone exists → force PHONE
    phones = extract_valid_phones_strict(s)
    if phones:
        print("Detected Possible PII: PHONE")
        print("Phones (E.164):", ", ".join(phones))
        print("-"*60)
        continue

    # Otherwise, predict with model
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_input)[0]
        for i in phone_idx: proba[i] = 0.0  # Hard-gate phone
        top_i = int(np.argmax(proba))
        final = classes[top_i] if proba[top_i] >= CONF_THRESHOLD else 'none'
    else:
        pred = model.predict(X_input)[0]
        final = 'none' if _is_phone_label(pred) else pred

    if _is_none_label(final):
        print("No sensitive information detected.")
    else:
        print(f"Detected Possible PII: {str(final).upper()}")
    print("-"*60)
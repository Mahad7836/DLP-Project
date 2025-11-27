# ----------------------------
#  AI + Regex Based Interactive DLP Detector (GLOBAL + HARD-GATED PHONE)
# ----------------------------
import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# ===== Global phone parsing (strict) =====
# pip install phonenumbers
import phonenumbers
from phonenumbers import PhoneNumberType, PhoneNumberFormat

ALLOWED_REGIONS = {"PK", "AE", "US", "GB", "DE", "IN"}   # edit for your markets
ALLOWED_TYPES   = {PhoneNumberType.MOBILE}               # mobiles only
DEFAULT_REGION  = "PK"                                   # interpret numbers without + as PK

CONTEXT_WORDS = {"call","phone","tel","mobile","mob","cell","whatsapp","wa","contact"}

def _near_context(text: str, i: int, j: int, win: int = 16) -> bool:
    if not text: return False
    lo = max(0, i-win); hi = min(len(text), j+win)
    ctx = text[lo:hi].lower()
    return any(w in ctx for w in CONTEXT_WORDS)

def _repetitive(ndigits: str) -> bool:
    return len(set(ndigits)) <= 2

def extract_valid_phones_strict(text: str, default_region: str = DEFAULT_REGION):
    """Return E.164 phones after strict multi-country checks."""
    if not text: return []
    out = []
    for m in phonenumbers.PhoneNumberMatcher(text, default_region):
        num = m.number
        if not (phonenumbers.is_possible_number(num) and phonenumbers.is_valid_number(num)):
            continue
        region = phonenumbers.region_code_for_number(num)
        if region and region not in ALLOWED_REGIONS:
            continue
        if phonenumbers.number_type(num) not in ALLOWED_TYPES:
            continue
        nd = str(num.national_number)
        if not (8 <= len(nd) <= 12):       # tune if needed
            continue
        if len(nd) == 6 or _repetitive(nd): # drop OTP-ish/junk patterns
            continue
        raw = m.raw_string.strip()
        if not raw.startswith('+') and not _near_context(text, m.start, m.end):
            continue
        out.append(phonenumbers.format_number(num, PhoneNumberFormat.E164))
    return out

def has_phone_strict(text: str) -> int:
    return int(bool(extract_valid_phones_strict(text)))

# ===== 1) Load data =====
df = pd.read_csv(r'data/dlp_data.csv')  # 'text','label'
df['text'] = df['text'].astype(str)
y = df['label'].astype(str)

# ===== 2) Regex features =====
EMAIL_RE = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
CNIC_RE  = re.compile(r'\b\d{5}-\d{7}-\d\b')

def has_email(t: str) -> int:
    return int(bool(EMAIL_RE.search(t or "")))

def has_cnic(t: str) -> int:
    return int(bool(CNIC_RE.search(t or "")))

df['email_flag'] = df['text'].apply(has_email)
df['phone_flag'] = df['text'].apply(has_phone_strict)  # strict, global
df['cnic_flag']  = df['text'].apply(has_cnic)

# ===== 3) TF-IDF =====
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
X_vec = vectorizer.fit_transform(df['text'])
regex_sparse = csr_matrix(df[['email_flag','phone_flag','cnic_flag']].values)
X = hstack([X_vec, regex_sparse])

# ===== 4) Split + Train =====
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_tr, y_tr)

# ===== 5) Eval =====
print("\nTraining Complete!")
print("\nModel Accuracy Report:\n")
print(classification_report(y_te, model.predict(X_te), digits=4))

# ----- helpers to handle label-name differences -----
def _canon(s: str) -> str:
    return re.sub(r'\s+', '', str(s or '').lower())

def _is_phone_label(name: str) -> bool:
    n = _canon(name)
    # match common variants: 'phone', 'phoneno', 'phonenumber', 'phones'
    return bool(re.search(r'phone', n))

def _is_none_label(name: str) -> bool:
    n = _canon(name)
    return n in {'none','no_pii','nopii','neutral','other','negative'}

# ===== 6) Interactive =====
print("\n--- Interactive PII Detection (Global, Hard-Gated) ---")
print("Type any text (email, phone, CNIC, etc.). Type 'exit' to quit.\n")

CONF_THRESHOLD = 0.65  # raise if you still see noise

classes = list(model.classes_)
canon_classes = [_canon(c) for c in classes]

# Indices for phone-like classes (case/variant insensitive)
phone_idx = [i for i,c in enumerate(classes) if _is_phone_label(c)]
none_idx  = [i for i,c in enumerate(classes) if _is_none_label(c)]

while True:
    s = input("Enter text to check: ")
    if s.lower().strip() == "exit":
        print("Exiting DLP Detector. Goodbye!")
        break

    # Strict signals
    phones = extract_valid_phones_strict(s)
    email_flag = has_email(s)
    phone_flag = int(bool(phones))  # ONLY 1 if strict phones exist
    cnic_flag  = has_cnic(s)

    X_text = vectorizer.transform([s])
    X_input = hstack([X_text, csr_matrix([[email_flag, phone_flag, cnic_flag]])])

    # If a real phone exists → PHONE, full stop
    if phones:
        final = 'phone'
        print("Detected Possible PII: PHONE")
        print("Phones (E.164):", ", ".join(phones))
        print("-" * 60)
        continue

    # Otherwise, get probabilities and zero out ALL phone-like classes
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_input)[0]
        if phone_idx:
            for i in phone_idx:
                proba[i] = 0.0  # HARD BLOCK: cannot predict phone without strict evidence

        # pick best remaining class
        top_i = int(np.argmax(proba))
        top_cls = classes[top_i]
        top_p   = float(proba[top_i])

        # confidence threshold → none
        if top_p < CONF_THRESHOLD:
            final = 'none'
        else:
            final = top_cls
    else:
        # No proba available: use raw prediction but remap phone->none when no evidence
        pred = model.predict(X_input)[0]
        final = 'none' if _is_phone_label(pred) else pred

    # Output
    if _is_none_label(final) or _canon(final) == 'none':
        print("No sensitive information detected.")
   print("-"*60)

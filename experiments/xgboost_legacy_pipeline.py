# file: models/pii_detector_b_shap.py
import re
import os
import unicodedata
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import phonenumbers
from phonenumbers import PhoneNumberType, PhoneNumberFormat

# SHAP + plotting
import shap
import matplotlib.pyplot as plt

# ----------------------------
# Settings
# ----------------------------
RND = 42
CONF_THRESHOLD = 0.65
ALLOWED_REGIONS = {"PK", "AE", "US", "GB", "DE", "IN"}
ALLOWED_TYPES = {PhoneNumberType.MOBILE}
DEFAULT_REGION = "PK"
CONTEXT_WORDS = {"call", "phone", "tel", "mobile", "mob", "cell", "whatsapp", "wa", "contact"}

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)
SHAP_PLOT_PATH = os.path.join(ARTIFACT_DIR, "shap_summary.png")

# ----------------------------
# Helpers: normalization, phone utilities
# ----------------------------
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
            if not (phonenumbers.is_possible_number(num) and phonenumbers.is_valid_number(num)):
                continue
            region = phonenumbers.region_code_for_number(num)
            if region and region not in ALLOWED_REGIONS:
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

# ----------------------------
# Regex detectors (CNIC removed)
# ----------------------------
EMAIL_RE = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')

# Address regex set (simple, catch-common patterns). Expand for your locale.
ADDRESS_PATTERNS = [
    r'\b(?:street|st\.?|road|rd\.?|avenue|ave\.?|lane|ln\.?|boulevard|blvd\.?|drive|dr\.?\b|sector|phase|block|house|flat|apt|apartment)\b',
    r'\b(?:po box|p\.?o\.?\s*box)\s*\d+\b',
    r'\b\d{1,5}\s+(?:[A-Za-z0-9]+\s){0,3}(?:street|st|road|rd|avenue|ave|lane|ln)\b',
    r'\bpostal\s*code\b',
    r'\bzip\s*code\b'
]
ADDRESS_RE = re.compile('|'.join(ADDRESS_PATTERNS), flags=re.IGNORECASE)

def has_email(text):
    return int(bool(EMAIL_RE.search(text or "")))

def has_address(text):
    return int(bool(ADDRESS_RE.search(text or "")))

# ----------------------------
# Load dataset & preprocess
# ----------------------------
DATA_PATH = r"data\newdata.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=['text', 'label']).copy()
df['text_norm'] = df['text'].apply(normalize_text)
df['text_lower'] = df['text_norm'].str.lower()

# ----------------------------
# Label encoding
# ----------------------------
label_encoder = LabelEncoder()
df['label_num'] = label_encoder.fit_transform(df['label'])
classes = label_encoder.classes_

# ----------------------------
# Feature flags (regex + phone) -- CNIC removed
# ----------------------------
df['email_flag'] = df['text_lower'].apply(has_email)
df['phone_flag'] = df['text_lower'].apply(has_phone_strict)
df['address_flag'] = df['text_lower'].apply(has_address)

# ----------------------------
# Train/test split
# ----------------------------
X_text = df['text_lower']
y = df['label_num']
X_tr_text, X_te_text, y_tr, y_te = train_test_split(X_text, y, test_size=0.3, stratify=y, random_state=RND)

# ----------------------------
# Vectorize (TF-IDF) and assemble feature matrices (include address_flag)
# ----------------------------
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=25000)
X_tr_vec = vectorizer.fit_transform(X_tr_text)
X_te_vec = vectorizer.transform(X_te_text)

# align regex features by index (CNIC removed)
regex_cols = ['email_flag', 'phone_flag', 'address_flag']
regex_tr = csr_matrix(df.loc[X_tr_text.index, regex_cols].values)
regex_te = csr_matrix(df.loc[X_te_text.index, regex_cols].values)

X_tr = hstack([X_tr_vec, regex_tr])
X_te = hstack([X_te_vec, regex_te])

# ----------------------------
# Train XGBoost classifier
# ----------------------------
model = xgb.XGBClassifier(
    n_estimators=600,
    max_depth=10,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    random_state=RND,
    n_jobs=-1,
    tree_method="hist"
)

print("Training XGBoost...")
model.fit(X_tr, y_tr)

# ----------------------------
# Evaluation
# ----------------------------
preds_num = model.predict(X_te)
preds_str = label_encoder.inverse_transform(preds_num)
y_te_str = label_encoder.inverse_transform(y_te.to_numpy())

print("\n=== XGBOOST PERFORMANCE ===")
print("Accuracy:", accuracy_score(y_te_str, preds_str))
print("\nClassification Report:\n", classification_report(y_te_str, preds_str, digits=4))
print("\nConfusion Matrix:\n", confusion_matrix(y_te_str, preds_str))

# ----------------------------
# Save artifacts (vectorizer, model, encoder)
# ----------------------------
joblib.dump(vectorizer,    os.path.join(ARTIFACT_DIR, "tfidf_vectorizer_xgb.joblib"))
joblib.dump(model,         os.path.join(ARTIFACT_DIR, "xgboost_classifier.joblib"))
joblib.dump(label_encoder, os.path.join(ARTIFACT_DIR, "label_encoder.joblib"))
joblib.dump(regex_cols,    os.path.join(ARTIFACT_DIR, "regex_feature_names.joblib"))
print("Artifacts saved.")

## ----------------------------
# SHAP explainability (TreeExplainer for XGBoost)
## ... (Keep previous code up to the "Save artifacts" section) ...

# ----------------------------
# SHAP explainability (Robust Fix)
# ----------------------------
# SHAP explainability (Generates plots for ALL labels)
# ----------------------------
print("Generating SHAP values...")
import matplotlib
matplotlib.use('Agg') # Essential for saving plots without a screen
import matplotlib.pyplot as plt

try:
    # 1. Sample (200 rows) to prevent memory crash
    SAMPLE_SIZE = 200
    if X_te.shape[0] > SAMPLE_SIZE:
        X_sample_sparse = X_te[:SAMPLE_SIZE]
    else:
        X_sample_sparse = X_te

    # 2. Compute SHAP values on the SPARSE matrix
    # check_additivity=False prevents floating point errors
    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer.shap_values(X_sample_sparse, check_additivity=False)

    # 3. Create Dense DataFrame just for Plotting (Matches Feature Names)
    feature_names = list(vectorizer.get_feature_names_out()) + ['email_flag', 'phone_flag', 'address_flag']
    X_sample_df = pd.DataFrame(X_sample_sparse.toarray(), columns=feature_names)

    # 4. Loop through EVERY class and generate a separate plot
    # classes_ usually contains ['ADDRESS', 'EMAIL', 'NONE', 'PHONE']
    for idx, class_label in enumerate(classes):
        print(f"Generating SHAP plot for class: {class_label}...")

        # -- SLICING LOGIC --
        # XGBoost+SHAP returns a 3D array (samples, features, classes) or a List of arrays.
        if isinstance(shap_values_raw, list):
            # Old SHAP version: List of [N, F] arrays
            shap_values_class = shap_values_raw[idx]
        elif len(shap_values_raw.shape) == 3:
            # New SHAP version: 3D Array [N, F, C] -> slice the 3rd dimension
            shap_values_class = shap_values_raw[:, :, idx]
        else:
            # Binary classification (only 2 classes)
            shap_values_class = shap_values_raw

        # -- PLOTTING --
        plt.figure() # Create fresh figure
        shap.summary_plot(
            shap_values_class,
            X_sample_df,
            show=False,
            plot_type="dot"
        )
        
        # Save file like "shap_summary_PHONE.png"
        filename = f"shap_summary_{str(class_label).upper()}.png"
        save_path = os.path.join(ARTIFACT_DIR, filename)
        
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close() # Clear memory
        print(f"Saved: {save_path}")

    print("All SHAP plots generated successfully.")

except Exception as e:
    print(f"SHAP generation failed: {e}")
# ----------------------------
# Interactive PII Detection (Uses regex gating for phone; address is soft-flag + model)
# ----------------------------
def _canon(s):
    return re.sub(r'\s+', '', str(s or '').lower())

def _is_phone_label(name):
    return bool(re.search(r'phone', _canon(name)))

def _is_none_label(name):
    return _canon(name) in {'none', 'no_pii', 'nopii', 'neutral', 'other', 'negative'}

# Identify indices for phone and none classes in the encoder classes
phone_idx = [i for i, c in enumerate(classes) if _is_phone_label(c)]
none_idx = [i for i, c in enumerate(classes) if _is_none_label(c)]

print("\n--- Interactive PII Detection ---")
print("Type any text (email, phone, address, etc.). Type 'exit' to quit.\n")

while True:
    s = input("Enter text to check: ")
    if s.lower().strip() in {"exit", "quit"}:
        print("Exiting DLP Detector. Goodbye!")
        break
    if not s.strip():
        continue

    # regex signals
    email_flag = has_email(s)
    phone_flag = has_phone_strict(s)
    address_flag = has_address(s)

    # vectorize input
    X_text_input = vectorizer.transform([s])
    X_input = hstack([X_text_input, csr_matrix([[email_flag, phone_flag, address_flag]])])

    # Hard-gate priority:
    # 1) Strict phone detection via phonenumbers → force PHONE
    phones = extract_valid_phones_strict(s)
    if phones:
        print("Detected Possible PII: PHONE (strict phone parser)")
        print("Phones (E.164):", ", ".join(phones))
        print("-" * 60)
        continue

    # Otherwise use model probabilities
    proba = model.predict_proba(X_input)[0]

    # Hard block: if model suggests phone but no strict regex evidence, zero-out phone probs
    for i in phone_idx:
        proba[i] = 0.0

    top_i = int(np.argmax(proba))
    confidence = float(proba[top_i])
    pred_label = classes[top_i]

    # If address_flag is present from regex, show a soft hint
    address_hint = bool(address_flag and not re.search(r'address|addr|location|place', pred_label.lower()))

    final = pred_label if confidence >= CONF_THRESHOLD else 'none'

    if _is_none_label(final):
        # If model is uncertain but address_flag is present, show potential address
        if address_flag:
            print("No high-confidence PII detected by model, but regex indicates potential ADDRESS.")
        else:
            print("No sensitive information detected.")
    else:
        print(f"Detected Possible PII: {str(final).upper()}")
        print(f"Confidence: {confidence:.2f}")
        if address_hint:
            print("Note: Address-like tokens were detected by regex (soft flag).")
    print("-" * 60)

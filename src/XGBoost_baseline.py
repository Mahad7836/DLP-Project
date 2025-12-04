import re
import pandas as pd
import numpy as np
import unicodedata
import os
import joblib
import optuna  # <--- NEW IMPORT
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
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
# 1. Load & Preprocess
# ----------------------------

print("Loading data...")
df = pd.read_csv(r"data\newdata.csv")
df = df.dropna(subset=['text', 'label'])

def normalize_text(s):
    s = unicodedata.normalize('NFKC', str(s))
    return s.strip()

df['text_norm'] = df['text'].apply(normalize_text)
df['text_lower'] = df['text_norm'].str.lower()

# Label Encoding
label_encoder = LabelEncoder()
df['label_num'] = label_encoder.fit_transform(df['label'])

X_text = df['text_lower']
y = df['label_num']

# ----------------------------
# 2. Feature Engineering
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

# Apply regex features
print("Extracting regex features...")
df['email_flag'] = df['text_lower'].apply(has_email)
df['cnic_flag'] = df['text_lower'].apply(has_cnic)
df['phone_flag'] = df['text_lower'].apply(has_phone_strict)

# Train/Test Split
X_tr_text, X_te_text, y_tr, y_te = train_test_split(
    X_text, y, test_size=0.3, stratify=y, random_state=RND
)
regex_tr_indices = X_tr_text.index
regex_te_indices = X_te_text.index

# TF-IDF Vectorization
print("Vectorizing text...")
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=25000)
X_tr_vec = vectorizer.fit_transform(X_tr_text)
X_te_vec = vectorizer.transform(X_te_text)

# Stack Features (TF-IDF + Regex Flags)
regex_tr = csr_matrix(df.loc[regex_tr_indices, ['email_flag', 'phone_flag', 'cnic_flag']].values)
regex_te = csr_matrix(df.loc[regex_te_indices, ['email_flag', 'phone_flag', 'cnic_flag']].values)

X_tr = hstack([X_tr_vec, regex_tr])
X_te = hstack([X_te_vec, regex_te])

# ----------------------------
# 3. OPTUNA OPTIMIZATION (THE UPGRADE)
# ----------------------------
print("\n--- Starting Optuna Optimization ---")
print("Searching for best hyperparameters (Running 5 trials)...")

def objective(trial):
    # This function is called 20 times to test different settings
    param = {
        'objective': 'multi:softmax', 
        'eval_metric': 'mlogloss',
        'tree_method': 'hist',
        'random_state': RND,
        'n_jobs': -1,
        'num_class': len(label_encoder.classes_),
        
        # Optuna will vary these values:
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    }

    model = xgb.XGBClassifier(**param)
    
    # We use Cross Validation (cv=3) to be sure the model is robust
    score = cross_val_score(model, X_tr, y_tr, cv=3, scoring='accuracy').mean()
    return score

# Run the optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=5) 

print("\n🎉 Best Hyperparameters found:")
print(study.best_params)

# ----------------------------
# 4. Train Final Model
# ----------------------------
print("\nTraining final model with best parameters...")

# Apply the best parameters found by Optuna
best_params = study.best_params
best_params['objective'] = 'multi:softprob' # Change to softprob for probabilities
best_params['tree_method'] = 'hist'
best_params['random_state'] = RND
best_params['n_jobs'] = -1

final_model = xgb.XGBClassifier(**best_params)
final_model.fit(X_tr, y_tr)

# ----------------------------
# 5. Evaluation & Saving
# ----------------------------
preds_num = final_model.predict(X_te)
preds_str = label_encoder.inverse_transform(preds_num)
y_te_str = label_encoder.inverse_transform(y_te)

print("\n=== OPTIMIZED XGBOOST PERFORMANCE ===")
print("Accuracy:", accuracy_score(y_te_str, preds_str))
print("\nClassification Report:\n", classification_report(y_te_str, preds_str, digits=4))

os.makedirs("artifacts", exist_ok=True)
joblib.dump(vectorizer,    "artifacts/tfidf_vectorizer_xgb.joblib")
joblib.dump(final_model,   "artifacts/xgboost_classifier.joblib")
joblib.dump(label_encoder, "artifacts/label_encoder.joblib")
print("Artifacts saved.")

# ----------------------------
# 6. Interactive Loop
# ----------------------------
def _canon(s):
    return re.sub(r'\s+', '', str(s or '').lower())

def _is_phone_label(name):
    return bool(re.search(r'phone', _canon(name)))

def _is_none_label(name):
    return _canon(name) in {'none', 'no_pii', 'nopii', 'neutral', 'other', 'negative'}

classes = label_encoder.classes_
phone_idx = [i for i, c in enumerate(classes) if _is_phone_label(c)]

print("\n" + "="*50)
print("🤖 OPTIMIZED MODEL INTERACTIVE TESTING")
print("Enter text to classify. Type 'quit' to stop.")
print("="*50)

while True:
    s = input("Enter text: ")
    if s.lower().strip() in ["exit", "quit"]:
        break

    if not s.strip():
        continue

    # Features
    email_flag = has_email(s)
    phone_flag = has_phone_strict(s)
    cnic_flag = has_cnic(s)

    # Transform
    X_text_input = vectorizer.transform([s])
    
    # --- IMPROVEMENT: Explicit Unknown Check ---
    if X_text_input.nnz == 0 and not (email_flag or phone_flag or cnic_flag):
        print(f"No sensitive information detected.\n")
        continue # Stops here, doesn't force a guess
    # -------------------------------------------

    X_input = hstack([X_text_input, csr_matrix([[email_flag, phone_flag, cnic_flag]])])

    # Strict Phone Check
    phones = extract_valid_phones_strict(s)
    if phones:
        print("Detected Possible PII: PHONE")
        print("Phones:", ", ".join(phones))
        print("-" * 50)
        continue

    # Predict
    proba = final_model.predict_proba(X_input)[0]
    
    # Hard-gate phone if regex failed
    for i in phone_idx:
        proba[i] = 0.0 

    top_i = int(np.argmax(proba))
    confidence = proba[top_i]
    pred_label = classes[top_i]

    final = pred_label if confidence >= CONF_THRESHOLD else 'none'

    if _is_none_label(final):
        print("No sensitive information detected.")
    else:
        print(f"Detected Possible PII: {str(final).upper()}")
        print(f"Confidence: {confidence:.2f}")
    
    print("-" * 50)

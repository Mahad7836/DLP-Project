import pandas as pd, re, unicodedata, os, joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

RND = 42

# =====================
# 1) Load CSV
# =====================
df = pd.read_csv(r"C:/Users/yashf/Desktop/DLP-Project/data/newdata.csv")

# Drop rows missing text or label
df = df.dropna(subset=['text', 'label'])

# =====================
# 2) Minimal normalization
# =====================
def normalize_text(s):
    s = unicodedata.normalize('NFKC', str(s))
    return s.strip()

df['text_norm']  = df['text'].apply(normalize_text)
df['text_lower'] = df['text_norm'].str.lower()

# =====================
# 3) LABEL ENCODING (REQUIRED FOR XGBOOST)
# =====================
label_encoder = LabelEncoder()
df['label_num'] = label_encoder.fit_transform(df['label'])

# Features + Numeric Labels
X = df['text_lower']
y = df['label_num']

# =====================
# 4) Train-test split
# =====================
X_tr_text, X_te_text, y_tr, y_te = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=RND
)

# =====================
# 5) TF-IDF Vectorizer
# =====================
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1,2),
    max_features=25000
)

X_tr_vec = vectorizer.fit_transform(X_tr_text)
X_te_vec = vectorizer.transform(X_te_text)

# =====================
# 6) XGBoost Model
# =====================
xgb_model = xgb.XGBClassifier(
    n_estimators=600,
    max_depth=10,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",   # XGB needs numeric labels
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=RND,
    n_jobs=-1
)

# Train
print("\nTraining XGBoost model... (this may take a minute)")
xgb_model.fit(X_tr_vec, y_tr)

# =====================
# 7) Predictions
# =====================
preds = xgb_model.predict(X_te_vec)

# Decode predictions back to original text labels
preds_decoded = label_encoder.inverse_transform(preds)
y_te_decoded = label_encoder.inverse_transform(y_te)

print("\n=== XGBOOST CLASSIFIER PERFORMANCE ===")
print("Accuracy:", accuracy_score(y_te_decoded, preds_decoded))
print("\nClassification Report:\n")
print(classification_report(y_te_decoded, preds_decoded, digits=4))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_te_decoded, preds_decoded))

# =====================
# 8) Save Artifacts
# =====================
os.makedirs("artifacts_xgb", exist_ok=True)

joblib.dump(vectorizer,      "artifacts_xgb/tfidf_vectorizer_xgb.joblib")
joblib.dump(xgb_model,       "artifacts_xgb/xgboost_classifier.joblib")
joblib.dump(label_encoder,   "artifacts_xgb/label_encoder.joblib")

print("\nXGBoost model training complete")
print("Artifacts saved in folder: artifacts_xgb/")

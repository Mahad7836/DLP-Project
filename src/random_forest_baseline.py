import pandas as pd, re, unicodedata, os, joblib
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

RND = 42
import pandas as pd
from sklearn.model_selection import train_test_split

# Load CSV safely using raw string for the Windows path
df = pd.read_csv(r"data/newdata.csv")

# Drop rows with NaN in either 'text' or 'label' columns
df = df.dropna(subset=['text', 'label'])  # replace 'label' with your target column name

# Split into features and labels
X = df['text']
y = df['label']

# Train-test split
X_tr_text, X_te_text, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)


# =====================
# 3) KEEP the NONE class (important!)
# =====================
# DO NOT remove "none"

# =====================
# 4) Minimal normalization
# =====================
def normalize_text(s):
    s = unicodedata.normalize('NFKC', str(s))
    return s.strip()

df['text_norm']  = df['text'].apply(normalize_text)
df['text_lower'] = df['text_norm'].str.lower()

# =====================
# 5) Split dataset
# =====================
X = df['text_lower']
y = df['label']

X_tr_text, X_te_text, y_tr, y_te = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=RND
)

# =====================
# 6) TF-IDF Vectorizer (no leakage)
# =====================
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1,2),
    max_features=25000
)

X_tr_vec = vectorizer.fit_transform(X_tr_text)
X_te_vec = vectorizer.transform(X_te_text)

# =====================
# 7) NO regex flags → remove leakage
# =====================
X_tr = X_tr_vec
X_te = X_te_vec

# =====================
# 8) Random Forest (clean)
# =====================
model = RandomForestClassifier(
    n_estimators=600,
    max_depth=None,
    class_weight="balanced_subsample",
    random_state=RND,
    n_jobs=-1
)

model.fit(X_tr, y_tr)
preds = model.predict(X_te)

print("\n=== FIXED RANDOM FOREST PERFORMANCE ===")
print("Accuracy:", accuracy_score(y_te, preds))
print("\nClassification Report:\n")
print(classification_report(y_te, preds, digits=4))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_te, preds))

# =====================
# 9) Save Artifacts
# =====================
os.makedirs("artifacts", exist_ok=True)

joblib.dump(vectorizer, "artifacts/tfidf_vectorizer.joblib")
joblib.dump(model, "artifacts/random_forest_fixed.joblib")

print("\nModel training complete. (Leakage-free)")
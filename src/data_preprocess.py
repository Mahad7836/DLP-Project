import pandas as pd, re, unicodedata, os, joblib
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer

RND = 42

# 1) load
df = pd.read_csv("duplicate.csv")
df['text'] = df['text'].astype(str)

# 2) drop exact duplicates
df = df.drop_duplicates(subset=['text','label']).reset_index(drop=True)

# 3) optional: drop tiny 'none' class
df = df[df['label'] != 'none'].reset_index(drop=True)

# 4) regex flags
#EMAIL_RE = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
#PHONE_RE = re.compile(r'(\+?\d[\d\-\(\)\s]{6,}\d)')
#CNIC_RE = re.compile(r'\b\d{5}-\d{7}-\d\b')

#df['email_flag'] = df['text'].str.contains(EMAIL_RE)
#df['phone_flag'] = df['text'].str.contains(PHONE_RE)
#df['cnic_flag'] = df['text'].str.contains(CNIC_RE)

# 5) normalize minimal
def normalize_text(s):
    s = unicodedata.normalize('NFKC', str(s))
    return s.strip()
df['text_norm'] = df['text'].apply(normalize_text)
df['text_lower'] = df['text_norm'].str.lower()

# 6) train/test split (stratified)
X = df['text_lower']
y = df['label']
X_tr_text, X_te_text, y_tr, y_te = train_test_split(X, y, test_size=0.30, stratify=y, random_state=RND)

# 7) vectorize on train only
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=20000)
X_tr_vec = vectorizer.fit_transform(X_tr_text)
X_te_vec = vectorizer.transform(X_te_text)

# 8) attach regex features aligned
train_meta = df.loc[X_tr_text.index, ['email_flag','phone_flag','cnic_flag']].values
test_meta  = df.loc[X_te_text.index, ['email_flag','phone_flag','cnic_flag']].values

X_tr = hstack([X_tr_vec, csr_matrix(train_meta)])
X_te = hstack([X_te_vec, csr_matrix(test_meta)])

# 9) save artifacts
os.makedirs('artifacts', exist_ok=True)
joblib.dump(vectorizer, 'artifacts/tfidf_vectorizer.joblib')
joblib.dump(['email_flag','phone_flag','cnic_flag'], 'artifacts/regex_features.joblib')
joblib.dump(list(vectorizer.get_feature_names_out()) + ['email_flag','phone_flag','cnic_flag'], 'artifacts/feature_names.joblib')

# 10) save updated dataframe to CSV
df.to_csv("duplicate_cleaned.csv", index=False)
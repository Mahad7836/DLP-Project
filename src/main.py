# ----------------------------
#  AI + Regex Based Interactive DLP Detector
# ----------------------------
import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# ----------------------------
#  1. Load dataset
# ----------------------------
df = pd.read_csv('dlp_dataset.csv')  # Make sure it has 'text' and 'label' columns

# ----------------------------
#  2. Define regex feature functions
# ----------------------------
def has_email(text):
    return int(bool(re.search(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', text)))

def has_phone(text):
    return int(bool(re.search(r'(\+92|0)?3\d{9}', text)))  # Pakistani format example

def has_cnic(text):
    return int(bool(re.search(r'\d{5}-\d{7}-\d', text)))

# Add regex feature columns
df['email_flag'] = df['text'].apply(has_email)
df['phone_flag'] = df['text'].apply(has_phone)
df['cnic_flag'] = df['text'].apply(has_cnic)

# ----------------------------
#  3. TF-IDF vectorization
# ----------------------------
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
X_vec = vectorizer.fit_transform(df['text'])

# Combine regex features with TF-IDF
from scipy.sparse import hstack
regex_features = df[['email_flag', 'phone_flag', 'cnic_flag']].values
X_combined = hstack([X_vec, regex_features])

y = df['label']

# Split for model training
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.3, random_state=42)

# ----------------------------
#  4. Train AI Model
# ----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ----------------------------
#  5. Quick Evaluation
# ----------------------------
print("\nTraining Complete!")
y_pred = model.predict(X_test)
print("\nModel Accuracy Report:\n")
print(classification_report(y_test, y_pred))

# ----------------------------
#  6. Interactive Testing Mode
# ----------------------------
print("\n--- Interactive PII Detection ---")
print("Type any text (email, phone, CNIC, etc.). Type 'exit' to quit.\n")

while True:
    user_input = input("Enter text to check: ")
    if user_input.lower() == "exit":
        print("Exiting DLP Detector. Goodbye!")
        break

    # Generate regex flags for this input
    email_flag = has_email(user_input)
    phone_flag = has_phone(user_input)
    cnic_flag = has_cnic(user_input)

    # Create dataframe for this single input
    temp_df = pd.DataFrame({
        "text": [user_input],
        "email_flag": [email_flag],
        "phone_flag": [phone_flag],
        "cnic_flag": [cnic_flag]
    })

    # Transform using same TF-IDF vectorizer
    X_text = vectorizer.transform(temp_df["text"])
    X_input = hstack([X_text, temp_df[["email_flag", "phone_flag", "cnic_flag"]].values])

    # Predict with trained model
    pred_label = model.predict(X_input)[0]

    # Determine detection status
    if pred_label != "none":
        print(f"Detected Possible PII: {pred_label.upper()}")
    else:
        print("No sensitive information detected.")
   print("-"*60)
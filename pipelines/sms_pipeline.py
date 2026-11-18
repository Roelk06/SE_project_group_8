import os
import re
import csv
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz, hstack, csr_matrix
from pathlib import Path


base_dir = Path(__file__).resolve().parent.parent
raw_file = base_dir / "data" / "raw" / "SMS" / "SMSSpamCollection"
processed_dir = base_dir / "data" / "processed" / "SMS"


def normalize_sms(text):
    # shortens stretched words: "FREEEEE" -> "FREE", "sooooo" -> "soo",
    # so the TF-IDF knows the word instead of skipping it as unknown
    return re.sub(r"(.)\1{2,}", r"\1\1", str(text))


def extra_features(text):
    # signals that the TF-IDF cannot see, because it ignores capitals and punctuation.
    # every value is between 0 and 1, like the TF-IDF values
    text = str(text)
    letters = [c for c in text if c.isalpha()]
    caps_ratio = sum(c.isupper() for c in letters) / len(letters) if letters else 0.0
    return [
        caps_ratio,                                                       # share of capital letters
        min(text.count("!"), 5) / 5,                                      # number of !
        min(text.count("?"), 5) / 5,                                      # number of ?
        float(bool(re.search(r"\d{5,}", text.replace(" ", "")))),        # phone or short code
        float(bool(re.search(r"[£$€]\s?\d|\d\s?(?:£|\$|€|euro|pounds?|dollars?)\b", text, re.IGNORECASE))),  # money amount
        float(bool(re.search(r"https?://|www\.|\.(?:com|net|org|co\.uk|nl)\b", text, re.IGNORECASE))),       # link
        float(bool(re.search(r"([a-zA-Z])\1{2,}", text))),               # stretched word like FREEEEE
    ]


def build_features(vectorizer, texts):
    # TF-IDF of the normalized text + the extra features of the original text
    tfidf = vectorizer.transform([normalize_sms(t) for t in texts])
    extra = csr_matrix(np.array([extra_features(t) for t in texts]))
    return hstack([tfidf, extra]).tocsr()


def preprocess_sms_input(text):
    vectorizer = joblib.load(processed_dir / "tfidf_vectorizer.pkl")
    return build_features(vectorizer, [text])

if __name__ == "__main__":
    processed_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(raw_file, sep="\t", header=None, names=["label", "text"], quoting=csv.QUOTE_NONE)
    data = data.drop_duplicates(subset="text").copy()

    labels_map = {"ham": 0, "spam": 1}
    data["label"] = data["label"].map(labels_map)

    X = data["text"]
    y = data["label"].to_numpy()

    X_train, X_temp, y_train, y_temp = train_test_split(X,y,test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    tfidf.fit([normalize_sms(t) for t in X_train])

    X_train_tfidf = build_features(tfidf, X_train)
    X_val_tfidf = build_features(tfidf, X_val)
    X_test_tfidf = build_features(tfidf, X_test)

    save_npz(os.path.join(processed_dir, "x_train.npz"), X_train_tfidf)
    save_npz(os.path.join(processed_dir, "x_val.npz"),X_val_tfidf)
    save_npz(os.path.join(processed_dir, "x_test.npz"),X_test_tfidf)
    np.save(os.path.join(processed_dir, "y_train.npy"),y_train)
    np.save(os.path.join(processed_dir, "y_val.npy"),y_val)
    np.save(os.path.join(processed_dir, "y_test.npy"),y_test)
    joblib.dump(tfidf,os.path.join(processed_dir, "tfidf_vectorizer.pkl"))

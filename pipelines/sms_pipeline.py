import os
import csv
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
from pathlib import Path


base_dir = Path(__file__).resolve().parent.parent
raw_file = base_dir / "data" / "raw" / "SMS" / "SMSSpamCollection"
processed_dir = base_dir / "data" / "processed" / "SMS"
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

X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)
X_test_tfidf = tfidf.transform(X_test)

save_npz(os.path.join(processed_dir, "x_train.npz"), X_train_tfidf)
save_npz(os.path.join(processed_dir, "x_val.npz"),X_val_tfidf)
save_npz(os.path.join(processed_dir, "x_test.npz"),X_test_tfidf)
np.save(os.path.join(processed_dir, "y_train.npy"),y_train)
np.save(os.path.join(processed_dir, "y_val.npy"),y_val)
np.save(os.path.join(processed_dir, "y_test.npy"),y_test)
joblib.dump(tfidf,os.path.join(processed_dir, "tfidf_vectorizer.pkl"))


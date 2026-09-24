import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
from pathlib import Path

base_dir = Path(__file__).resolve().parent
raw_file = base_dir /'data'/'raw'/'emails'/'enron_spam_data.csv'
processed_dir = base_dir /'data'/'processed'/'emails'
processed_dir.mkdir(parents=True, exist_ok=True)

data = pd.read_csv(raw_file)
data['text'] = data["Subject"].fillna('') + ' ' + data["Message"].fillna('')
data = data.drop_duplicates(subset=['text']).copy()

labels_map = {'ham': 0, 'spam': 1}
data['label'] = data['Spam/Ham'].map(labels_map)
data = data.dropna(subset=['label'])

x = data['text']
y = data['label'].to_numpy(dtype = int)

x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, stratify=y, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')

x_train_tfidf = tfidf.fit_transform(x_train)
x_val_tfidf = tfidf.transform(x_val)
x_test_tfidf = tfidf.transform(x_test)

save_npz(os.path.join(processed_dir, 'x_train.npz'), x_train_tfidf)
save_npz(os.path.join(processed_dir, 'x_val.npz'), x_val_tfidf)
save_npz(os.path.join(processed_dir, 'x_test.npz'), x_test_tfidf)
np.save(os.path.join(processed_dir, 'y_train.npy'), y_train)
np.save(os.path.join(processed_dir, 'y_val.npy'), y_val)
np.save(os.path.join(processed_dir, 'y_test.npy'), y_test)
joblib.dump(tfidf, os.path.join(processed_dir, 'tfidf_vectorizer.pkl'))


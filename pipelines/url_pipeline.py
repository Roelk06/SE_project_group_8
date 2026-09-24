import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
raw_file = base_dir /'data'/'raw'/'urls'/'phishing_site_urls.csv'
processed_dir = base_dir /'data'/'processed'/'urls'
processed_dir.mkdir(parents=True, exist_ok=True)

data = pd.read_csv(raw_file)
data = data.drop_duplicates(subset=["URL"]).copy()

labels_map = {'good': 0, 'bad': 1}
data['label'] = data['Label'].map(labels_map)
data = data.dropna(subset=['label'])

x = data['URL'].astype(str)
y = data['label'].to_numpy(dtype=int)

x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, stratify=y, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

char_tfidf = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=5000)

x_train_tfidf = char_tfidf.fit_transform(x_train)
x_val_tfidf = char_tfidf.transform(x_val)
x_test_tfidf = char_tfidf.transform(x_test)

save_npz(os.path.join(processed_dir, 'x_train.npz'), x_train_tfidf)
save_npz(os.path.join(processed_dir, 'x_val.npz'), x_val_tfidf)
save_npz(os.path.join(processed_dir, 'x_test.npz'), x_test_tfidf)
np.save(os.path.join(processed_dir, 'y_train.npy'), y_train)
np.save(os.path.join(processed_dir, 'y_val.npy'), y_val)
np.save(os.path.join(processed_dir, 'y_test.npy'), y_test)
joblib.dump(char_tfidf, os.path.join(processed_dir, 'char_tfidf_vectorizer.pkl'))


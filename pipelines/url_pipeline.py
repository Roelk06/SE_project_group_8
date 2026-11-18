import os
import re
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
from pathlib import Path
from urllib.parse import urlparse

base_dir = Path(__file__).resolve().parent.parent
raw_file = base_dir /'data'/'raw'/'urls'/'phishing_site_urls.csv'
processed_dir = base_dir /'data'/'processed'/'urls'

# Well-known websites that are always classified as legitimate, without asking the model.
# The model only reads the letters of a URL, so it cannot tell the real google.com from
# a fake page that mentions "google". Subdomains count too (docs.google.com, en.wikipedia.org).
# Downside: phishing pages hosted on these sites (for example a fake Google Form) are let through.
TRUSTED_DOMAINS = {
    "google.com", "youtube.com", "gmail.com", "microsoft.com", "live.com", "outlook.com",
    "office.com", "apple.com", "icloud.com", "amazon.com", "paypal.com", "facebook.com",
    "instagram.com", "whatsapp.com", "linkedin.com", "x.com", "twitter.com", "github.com",
    "wikipedia.org", "reddit.com", "netflix.com", "spotify.com", "dropbox.com", "zoom.us",
    "bol.com",
}


def normalize_url(url):
    # The training URLs have no "http://" or "https://" and few "www.", while real
    # links almost always have them. Removing them makes both look the same.
    url = str(url).strip()
    url = re.sub(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", "", url)
    url = re.sub(r"^www\.", "", url, flags=re.IGNORECASE)
    return url


def get_domain(url):
    url = str(url).strip()
    host = urlparse(url if "://" in url else "//" + url).hostname or ""
    host = host.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def is_trusted_domain(url):
    domain = get_domain(url)
    return any(domain == trusted or domain.endswith("." + trusted) for trusted in TRUSTED_DOMAINS)


def preprocess_url_input(url):
    vectorizer = joblib.load(processed_dir / "char_tfidf_vectorizer.pkl")
    return vectorizer.transform([normalize_url(url)])

if __name__ == "__main__":
    processed_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(raw_file)
    data['URL'] = data['URL'].astype(str).map(normalize_url)
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

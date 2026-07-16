import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from nltk.stem import SnowballStemmer
import time

DATA_DIR = "C:/Users/PC/Downloads/avito/candidate_data"
stemmer = SnowballStemmer("russian")

def clean_html(html_text):
    if not html_text or not isinstance(html_text, str):
        return ""
    soup = BeautifulSoup(html_text, "html.parser")
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-zа-яё0-9\s]", " ", text)
    return text.split()

def tokenize_stemmed(text):
    text = text.lower()
    text = re.sub(r"[^a-zа-яё0-9\s]", " ", text)
    return [stemmer.stem(t) for t in text.split()]

def map_at_k(predicted, relevant, k=10):
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    score = 0.0
    hits = 0
    for i, pid in enumerate(predicted[:k]):
        if pid in relevant_set:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(relevant_set), k)

def rrf(rankings, weights=None, k=60):
    if weights is None:
        weights = [1.0] * len(rankings)
    scores = defaultdict(float)
    for ranking, w in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] += w / (k + rank + 1)
    return [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])]

t0 = time.time()
print("Loading data...")
articles = pd.read_feather(f"{DATA_DIR}/articles.f")
calibration = pd.read_feather(f"{DATA_DIR}/calibration.f")
test = pd.read_feather(f"{DATA_DIR}/test.f")

print("Cleaning HTML...")
articles["clean_body"] = articles["body"].apply(clean_html)
articles["full_text"] = (articles["title"] + " ") * 3 + articles["clean_body"]
articles["tokens"] = articles["full_text"].apply(tokenize)
articles["tokens_stem"] = articles["full_text"].apply(tokenize_stemmed)

article_ids = articles["article_id"].tolist()
N = len(articles)
print(f"Articles: {N}, Calibration: {len(calibration)}, Test: {len(test)}")
print(f"Data loaded in {time.time()-t0:.1f}s")

t1 = time.time()
print("Building indices...")
bm25 = BM25Okapi(articles["tokens"].tolist())
bm25_stem = BM25Okapi(articles["tokens_stem"].tolist())
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(articles["full_text"].tolist())

print("Loading MiniLM + encoding articles...")
encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
art_emb = encoder.encode(articles["full_text"].tolist(), batch_size=128, show_progress_bar=False, normalize_embeddings=True)
print(f"Indices built in {time.time()-t1:.1f}s")

def rank_query(query, top_k=10, candidate_k=50):
    tokens = tokenize(query)
    tokens_st = tokenize_stemmed(query)
    
    bm25_scores = bm25.get_scores(tokens)
    bm25_st_scores = bm25_stem.get_scores(tokens_st)
    tfidf_scores = (tfidf_matrix @ tfidf.transform([query]).T).toarray().flatten()
    q_emb = encoder.encode([query], show_progress_bar=False, normalize_embeddings=True)[0]
    emb_scores = art_emb @ q_emb
    
    idx_bm25 = np.argsort(bm25_scores)[::-1][:candidate_k]
    idx_stem = np.argsort(bm25_st_scores)[::-1][:candidate_k]
    idx_tfidf = np.argsort(tfidf_scores)[::-1][:candidate_k]
    idx_emb = np.argsort(emb_scores)[::-1][:candidate_k]
    
    r1 = [article_ids[i] for i in idx_bm25]
    r2 = [article_ids[i] for i in idx_stem]
    r3 = [article_ids[i] for i in idx_tfidf]
    r4 = [article_ids[i] for i in idx_emb]
    
    fused = rrf([r1, r2, r3, r4], weights=[1.0, 0.8, 1.0, 1.5], k=60)
    return fused[:top_k]

print("Validating on calibration...")
t2 = time.time()
ap_scores = []
for _, row in calibration.iterrows():
    predicted = rank_query(row["query_text"], top_k=10)
    relevant = list(map(int, row["ground_truth"].split()))
    ap_scores.append(map_at_k(predicted, relevant))
print(f"MAP@10 = {np.mean(ap_scores):.4f} ({time.time()-t2:.1f}s)")

print("Generating test predictions...")
t3 = time.time()
results = []
for _, row in test.iterrows():
    predicted = rank_query(row["query_text"], top_k=10)
    results.append({"query_id": row["query_id"], "answer": " ".join(map(str, predicted))})

pd.DataFrame(results).to_csv("C:/Users/PC/Downloads/avito/answer.csv", index=False)
print(f"Done in {time.time()-t3:.1f}s. Total: {time.time()-t0:.1f}s")

import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import json

DATA_DIR = "C:/Users/PC/Downloads/avito/candidate_data"
stemmer = SnowballStemmer("russian")
russian_stopwords = set(stopwords.words("russian"))

def clean_html(html_text):
    if not html_text or not isinstance(html_text, str):
        return ""
    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text, use_stopwords=False):
    text = text.lower()
    text = re.sub(r"[^a-zа-яё0-9\s]", " ", text)
    tokens = text.split()
    if use_stopwords:
        tokens = [t for t in tokens if t not in russian_stopwords and len(t) > 2]
    return tokens

def tokenize_stemmed(text, use_stopwords=False):
    text = text.lower()
    text = re.sub(r"[^a-zа-яё0-9\s]", " ", text)
    tokens = text.split()
    stemmed = [stemmer.stem(t) for t in tokens if t not in russian_stopwords or not use_stopwords]
    if use_stopwords:
        stemmed = [t for t in stemmed if len(t) > 2]
    return stemmed

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

print("Loading data...")
articles = pd.read_feather(f"{DATA_DIR}/articles.f")
calibration = pd.read_feather(f"{DATA_DIR}/calibration.f")
test = pd.read_feather(f"{DATA_DIR}/test.f")

print("Cleaning HTML...")
articles["clean_body"] = articles["body"].apply(clean_html)
articles["title_text"] = articles["title"].fillna("")
articles["full_text"] = (articles["title_text"] + " ") * 3 + articles["clean_body"]

articles["tokens"] = articles["full_text"].apply(lambda x: tokenize(x, False))
articles["tokens_stem"] = articles["full_text"].apply(lambda x: tokenize_stemmed(x, False))
articles["tokens_stop"] = articles["full_text"].apply(lambda x: tokenize(x, True))
articles["title_tokens"] = articles["title_text"].apply(lambda x: tokenize(x, False))
articles["title_tokens_stop"] = articles["title_text"].apply(lambda x: tokenize(x, True))

article_ids = articles["article_id"].tolist()

print("Building indices...")
bm25 = BM25Okapi(articles["tokens"].tolist())
bm25_stem = BM25Okapi(articles["tokens_stem"].tolist())
bm25_stop = BM25Okapi(articles["tokens_stop"].tolist())
bm25_title = BM25Okapi(articles["title_tokens"].tolist())
bm25_title_stop = BM25Okapi(articles["title_tokens_stop"].tolist())

tfidf = TfidfVectorizer(ngram_range=(1, 2), max_df=0.95, min_df=1)
tfidf_matrix = tfidf.fit_transform(articles["full_text"].tolist())
tfidf_title = TfidfVectorizer(ngram_range=(1, 2), max_df=0.95, min_df=1)
tfidf_title_matrix = tfidf_title.fit_transform(articles["title_text"].tolist())

print("Loading embeddings...")
encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
art_emb = encoder.encode(articles["full_text"].tolist(), batch_size=128, show_progress_bar=False, normalize_embeddings=True)
art_title_emb = encoder.encode(articles["title_text"].tolist(), batch_size=128, show_progress_bar=False, normalize_embeddings=True)

cal_queries = calibration["query_text"].tolist()
cal_relevant = [list(map(int, x.split())) for x in calibration["ground_truth"].tolist()]

def get_rankings(query, candidate_k=50):
    tokens = tokenize(query, False)
    tokens_st = tokenize_stemmed(query, False)
    tokens_stop = tokenize(query, True)
    title_tokens = tokenize(query, False)
    title_tokens_stop = tokenize(query, True)
    
    bm25_scores = bm25.get_scores(tokens)
    bm25_st_scores = bm25_stem.get_scores(tokens_st)
    bm25_stop_scores = bm25_stop.get_scores(tokens_stop)
    bm25_title_scores = bm25_title.get_scores(title_tokens)
    bm25_title_stop_scores = bm25_title_stop.get_scores(title_tokens_stop)
    tfidf_scores = (tfidf_matrix @ tfidf.transform([query]).T).toarray().flatten()
    tfidf_title_scores = (tfidf_title_matrix @ tfidf_title.transform([query]).T).toarray().flatten()
    q_emb = encoder.encode([query], show_progress_bar=False, normalize_embeddings=True)[0]
    emb_scores = art_emb @ q_emb
    title_scores = art_title_emb @ q_emb
    
    def top_ids(scores):
        return [article_ids[i] for i in np.argsort(scores)[::-1][:candidate_k]]
    
    return [
        top_ids(bm25_scores),
        top_ids(bm25_st_scores),
        top_ids(bm25_stop_scores),
        top_ids(bm25_title_scores),
        top_ids(bm25_title_stop_scores),
        top_ids(tfidf_scores),
        top_ids(tfidf_title_scores),
        top_ids(emb_scores),
        top_ids(title_scores),
    ]

print("Computing calibration rankings...")
cal_rankings = [get_rankings(q, candidate_k=50) for q in cal_queries]

print("Running grid search...")
best_map = 0
best_weights = None
best_k = None

weight_options = []
for emb_w in [1.5, 1.8, 2.0, 2.2, 2.5]:
    for title_emb_w in [0.5, 0.8, 1.0, 1.2, 1.5]:
        for tfidf_w in [0.8, 1.0, 1.2]:
            for title_tfidf_w in [0.5, 0.8, 1.0]:
                weight_options.append([1.0, 0.8, 0.8, 1.0, 0.8, tfidf_w, title_tfidf_w, emb_w, title_emb_w])

for k_val in [30, 40, 60]:
    for weights in weight_options:
        ap_scores = []
        for rankings, relevant in zip(cal_rankings, cal_relevant):
            predicted = rrf(rankings, weights=weights, k=k_val)[:10]
            ap_scores.append(map_at_k(predicted, relevant))
        map_score = np.mean(ap_scores)
        if map_score > best_map:
            best_map = map_score
            best_weights = weights
            best_k = k_val

print(f"Best MAP@10 = {best_map:.4f}")
print(f"Best weights = {best_weights}")
print(f"Best k = {best_k}")

config = {"weights": best_weights, "k": best_k, "map": best_map}
with open("C:/Users/PC/Downloads/avito/config.json", "w", encoding="utf-8") as f:
    json.dump(config, f)

def rank_query(query, top_k=10, candidate_k=50):
    rankings = get_rankings(query, candidate_k)
    return rrf(rankings, weights=best_weights, k=best_k)[:top_k]

print("Validating final on calibration...")
ap_scores = [map_at_k(rank_query(q, 10), rel) for q, rel in zip(cal_queries, cal_relevant)]
print(f"Final MAP@10 = {np.mean(ap_scores):.4f}")

print("Generating test predictions...")
results = []
for _, row in test.iterrows():
    predicted = rank_query(row["query_text"], top_k=10)
    results.append({"query_id": row["query_id"], "answer": " ".join(map(str, predicted))})

pd.DataFrame(results).to_csv("C:/Users/PC/Downloads/avito/answer.csv", index=False)
print("Saved answer.csv")

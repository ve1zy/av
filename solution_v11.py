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

article_ids = articles["article_id"].tolist()

cal_queries = calibration["query_text"].tolist()
cal_relevant = [list(map(int, x.split())) for x in calibration["ground_truth"].tolist()]

print("Loading embeddings...")
encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
art_emb = encoder.encode(articles["full_text"].tolist(), batch_size=128, show_progress_bar=False, normalize_embeddings=True)
art_title_emb = encoder.encode(articles["title_text"].tolist(), batch_size=128, show_progress_bar=False, normalize_embeddings=True)

print("Fine grid search BM25 + TF-IDF params...")
best_map = 0
best_config = None

# Финальные веса из v10
best_weights = [1.0, 1.0, 1.0, 1.0, 1.8, 0.8]

k1_values = [1.8, 2.0, 2.2, 2.5]
b_values = [0.3, 0.4, 0.5, 0.6]
ngram_options = [(1, 1), (1, 2), (1, 3)]
min_df_options = [1, 2]

for k1 in k1_values:
    for b in b_values:
        for ngram in ngram_options:
            for min_df in min_df_options:
                bm25 = BM25Okapi(articles["tokens"].tolist(), k1=k1, b=b)
                bm25_stem = BM25Okapi(articles["tokens_stem"].tolist(), k1=k1, b=b)
                bm25_stop = BM25Okapi(articles["tokens_stop"].tolist(), k1=k1, b=b)
                tfidf = TfidfVectorizer(ngram_range=ngram, max_df=0.95, min_df=min_df)
                tfidf_matrix = tfidf.fit_transform(articles["full_text"].tolist())
                
                def get_rankings(query, candidate_k=50):
                    tokens = tokenize(query, False)
                    tokens_st = tokenize_stemmed(query, False)
                    tokens_stop = tokenize(query, True)
                    
                    bm25_scores = bm25.get_scores(tokens)
                    bm25_st_scores = bm25_stem.get_scores(tokens_st)
                    bm25_stop_scores = bm25_stop.get_scores(tokens_stop)
                    tfidf_scores = (tfidf_matrix @ tfidf.transform([query]).T).toarray().flatten()
                    q_emb = encoder.encode([query], show_progress_bar=False, normalize_embeddings=True)[0]
                    emb_scores = art_emb @ q_emb
                    title_scores = art_title_emb @ q_emb
                    
                    def top_ids(scores):
                        return [article_ids[i] for i in np.argsort(scores)[::-1][:candidate_k]]
                    
                    return [
                        top_ids(bm25_scores),
                        top_ids(bm25_st_scores),
                        top_ids(bm25_stop_scores),
                        top_ids(tfidf_scores),
                        top_ids(emb_scores),
                        top_ids(title_scores),
                    ]
                
                cal_rankings = [get_rankings(q, candidate_k=50) for q in cal_queries]
                ap_scores = [map_at_k(rrf(r, weights=best_weights, k=30)[:10], rel) for r, rel in zip(cal_rankings, cal_relevant)]
                map_score = np.mean(ap_scores)
                
                if map_score > best_map:
                    best_map = map_score
                    best_config = {"bm25_k1": k1, "bm25_b": b, "ngram": ngram, "min_df": min_df, "map": map_score}

print(f"Best MAP@10 = {best_map:.4f}")
print(f"Best config = {best_config}")

# Пересоздаём финальные индексы
k1_best = best_config["bm25_k1"]
b_best = best_config["bm25_b"]
ngram_best = best_config["ngram"]
min_df_best = best_config["min_df"]

bm25 = BM25Okapi(articles["tokens"].tolist(), k1=k1_best, b=b_best)
bm25_stem = BM25Okapi(articles["tokens_stem"].tolist(), k1=k1_best, b=b_best)
bm25_stop = BM25Okapi(articles["tokens_stop"].tolist(), k1=k1_best, b=b_best)
tfidf = TfidfVectorizer(ngram_range=ngram_best, max_df=0.95, min_df=min_df_best)
tfidf_matrix = tfidf.fit_transform(articles["full_text"].tolist())

with open("C:/Users/PC/Downloads/avito/config.json", "w", encoding="utf-8") as f:
    json.dump({**best_config, "weights": best_weights, "k": 30}, f)

def get_rankings(query, candidate_k=50):
    tokens = tokenize(query, False)
    tokens_st = tokenize_stemmed(query, False)
    tokens_stop = tokenize(query, True)
    
    bm25_scores = bm25.get_scores(tokens)
    bm25_st_scores = bm25_stem.get_scores(tokens_st)
    bm25_stop_scores = bm25_stop.get_scores(tokens_stop)
    tfidf_scores = (tfidf_matrix @ tfidf.transform([query]).T).toarray().flatten()
    q_emb = encoder.encode([query], show_progress_bar=False, normalize_embeddings=True)[0]
    emb_scores = art_emb @ q_emb
    title_scores = art_title_emb @ q_emb
    
    def top_ids(scores):
        return [article_ids[i] for i in np.argsort(scores)[::-1][:candidate_k]]
    
    return [
        top_ids(bm25_scores),
        top_ids(bm25_st_scores),
        top_ids(bm25_stop_scores),
        top_ids(tfidf_scores),
        top_ids(emb_scores),
        top_ids(title_scores),
    ]

def rank_query(query, top_k=10, candidate_k=50):
    rankings = get_rankings(query, candidate_k)
    return rrf(rankings, weights=best_weights, k=30)[:top_k]

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

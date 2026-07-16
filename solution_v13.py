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
import lightgbm as lgb
import json
import time

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

article_ids = articles["article_id"].tolist()
id_to_idx = {aid: i for i, aid in enumerate(article_ids)}

print("Building BM25/TF-IDF indices...")
bm25 = BM25Okapi(articles["tokens"].tolist(), k1=2.0, b=0.5)
bm25_stem = BM25Okapi(articles["tokens_stem"].tolist(), k1=2.0, b=0.5)
bm25_stop = BM25Okapi(articles["tokens_stop"].tolist(), k1=2.0, b=0.5)
bm25_title = BM25Okapi(articles["title_tokens"].tolist(), k1=2.0, b=0.5)

tfidf = TfidfVectorizer(ngram_range=(1, 2), max_df=0.95, min_df=1)
tfidf_matrix = tfidf.fit_transform(articles["full_text"].tolist())
tfidf_title = TfidfVectorizer(ngram_range=(1, 2), max_df=0.95, min_df=1)
tfidf_title_matrix = tfidf_title.fit_transform(articles["title_text"].tolist())

print("Loading embedding models...")
models = {
    "minilm": SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2"),
    "rubert": SentenceTransformer("cointegrated/rubert-tiny2"),
}

print("Encoding articles...")
art_embs = {}
for name, model in models.items():
    print(f"  {name} full...")
    full_emb = model.encode(articles["full_text"].tolist(), batch_size=128, show_progress_bar=False, normalize_embeddings=True)
    print(f"  {name} title...")
    title_emb = model.encode(articles["title_text"].tolist(), batch_size=128, show_progress_bar=False, normalize_embeddings=True)
    art_embs[name] = (model, full_emb, title_emb)

# Предвычисляем scores для всех статей по каждому запросу
def get_all_scores(query):
    tokens = tokenize(query, False)
    tokens_st = tokenize_stemmed(query, False)
    tokens_stop = tokenize(query, True)
    title_tokens = tokenize(query, False)
    
    scores = {
        "bm25": bm25.get_scores(tokens),
        "bm25_stem": bm25_stem.get_scores(tokens_st),
        "bm25_stop": bm25_stop.get_scores(tokens_stop),
        "bm25_title": bm25_title.get_scores(title_tokens),
        "tfidf": (tfidf_matrix @ tfidf.transform([query]).T).toarray().flatten(),
        "tfidf_title": (tfidf_title_matrix @ tfidf_title.transform([query]).T).toarray().flatten(),
    }
    
    for name, (model, full_emb, title_emb) in art_embs.items():
        q_emb = model.encode([query], show_progress_bar=False, normalize_embeddings=True)[0]
        scores[f"emb_{name}"] = full_emb @ q_emb
        scores[f"emb_title_{name}"] = title_emb @ q_emb
    
    return scores

def build_features_for_candidates(query, candidate_ids):
    scores = get_all_scores(query)
    
    # Нормализуем scores в [0,1] для каждого ранкера
    normalized = {}
    for key, s in scores.items():
        smin, smax = s.min(), s.max()
        if smax > smin:
            normalized[key] = (s - smin) / (smax - smin)
        else:
            normalized[key] = np.zeros_like(s)
    
    # Ранги
    ranks = {key: np.argsort(np.argsort(-s)) for key, s in scores.items()}
    
    features = []
    for aid in candidate_ids:
        idx = id_to_idx[aid]
        f = []
        for key in sorted(scores.keys()):
            f.append(scores[key][idx])
            f.append(normalized[key][idx])
            f.append(ranks[key][idx])
        f.append(len(query))
        f.append(len(articles.iloc[idx]["full_text"]))
        f.append(len(articles.iloc[idx]["title_text"]))
        features.append(f)
    
    return np.array(features)

# Получаем кандидатов через RRF
def rrf(rankings, weights=None, k=60):
    if weights is None:
        weights = [1.0] * len(rankings)
    scores = defaultdict(float)
    for ranking, w in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] += w / (k + rank + 1)
    return [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])]

def get_candidates(query, candidate_k=50):
    scores = get_all_scores(query)
    
    rankings = []
    for key in scores.keys():
        rankings.append([article_ids[i] for i in np.argsort(scores[key])[::-1][:candidate_k]])
    
    # Веса под RRF из v10 + rubert
    weights = [1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 1.5, 0.8, 0.5, 0.3]
    return rrf(rankings, weights=weights, k=30)[:candidate_k]

print("Building training data...")
train_features = []
train_labels = []
for idx, row in calibration.iterrows():
    if idx % 100 == 0:
        print(f"  {idx}/{len(calibration)}")
    query = row["query_text"]
    relevant = set(map(int, row["ground_truth"].split()))
    
    candidates = get_candidates(query, candidate_k=50)
    features = build_features_for_candidates(query, candidates)
    labels = [1 if aid in relevant else 0 for aid in candidates]
    
    train_features.append(features)
    train_labels.extend(labels)

X_train = np.vstack(train_features)
y_train = np.array(train_labels)

print(f"Training data: {X_train.shape}, positives: {y_train.sum()}")

print("Training LightGBM...")
train_set = lgb.Dataset(X_train, label=y_train)
params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "scale_pos_weight": (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
}

ranker = lgb.train(params, train_set, num_boost_round=200)

# Важность фич
importance = pd.DataFrame({
    "feature": range(X_train.shape[1]),
    "importance": ranker.feature_importance(importance_type="gain")
}).sort_values("importance", ascending=False)
print("Top features:")
print(importance.head(15))

def rank_with_ltr(query, top_k=10):
    candidates = get_candidates(query, candidate_k=50)
    if not candidates:
        return []
    features = build_features_for_candidates(query, candidates)
    preds = ranker.predict(features)
    ranked_indices = np.argsort(preds)[::-1]
    return [candidates[i] for i in ranked_indices[:top_k]]

print("Validating on calibration...")
ap_scores = []
for idx, row in calibration.iterrows():
    if idx % 100 == 0:
        print(f"  {idx}/{len(calibration)}")
    predicted = rank_with_ltr(row["query_text"], top_k=10)
    relevant = list(map(int, row["ground_truth"].split()))
    ap_scores.append(map_at_k(predicted, relevant))

map_score = np.mean(ap_scores)
print(f"MAP@10 on calibration (LTR): {map_score:.4f}")

print("Generating test predictions...")
results = []
for idx, row in test.iterrows():
    if idx % 100 == 0:
        print(f"  {idx}/{len(test)}")
    predicted = rank_with_ltr(row["query_text"], top_k=10)
    results.append({"query_id": row["query_id"], "answer": " ".join(map(str, predicted))})

pd.DataFrame(results).to_csv("C:/Users/PC/Downloads/avito/answer.csv", index=False)
print("Saved answer.csv")

# Сохраняем модель
ranker.save_model("C:/Users/PC/Downloads/avito/lgbm_model.txt")
print("Saved lgbm_model.txt")

import pandas as pd
import numpy as np
import re
import os
from collections import Counter
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import lightgbm as lgb

DATA_DIR = "./candidate_data"
CACHE_DIR = "./cache2"
os.makedirs(CACHE_DIR, exist_ok=True)

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

def get_ngrams(tokens, n=2):
    return set(zip(*[tokens[i:] for i in range(n)]))

def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

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

article_ids = articles["article_id"].tolist()
id_to_idx = {aid: i for i, aid in enumerate(article_ids)}

print("Building indices...")
bm25 = BM25Okapi(articles["tokens"].tolist(), k1=2.0, b=0.5)
bm25_stem = BM25Okapi(articles["tokens_stem"].tolist(), k1=2.0, b=0.5)
bm25_stop = BM25Okapi(articles["tokens_stop"].tolist(), k1=2.0, b=0.5)
bm25_title = BM25Okapi(articles["title_tokens"].tolist(), k1=2.0, b=0.5)

tfidf = TfidfVectorizer(ngram_range=(1, 2), max_df=0.95, min_df=1)
tfidf_matrix = tfidf.fit_transform(articles["full_text"].tolist())
tfidf_title = TfidfVectorizer(ngram_range=(1, 2), max_df=0.95, min_df=1)
tfidf_title_matrix = tfidf_title.fit_transform(articles["title_text"].tolist())

# Кэширование embeddings (batch encoding - быстро)
def get_or_encode(cache_path, texts, model_name, batch_size=128):
    if os.path.exists(cache_path):
        return np.load(cache_path)
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
    np.save(cache_path, emb)
    return emb

print("Encoding MiniLM embeddings (cached)...")
minilm_full = get_or_encode(f"{CACHE_DIR}/minilm_full.npy", articles["full_text"].tolist(), "paraphrase-multilingual-MiniLM-L12-v2")
minilm_title = get_or_encode(f"{CACHE_DIR}/minilm_title.npy", articles["title_text"].tolist(), "paraphrase-multilingual-MiniLM-L12-v2")
minilm_cal_q = get_or_encode(f"{CACHE_DIR}/minilm_cal_q.npy", calibration["query_text"].tolist(), "paraphrase-multilingual-MiniLM-L12-v2")
minilm_test_q = get_or_encode(f"{CACHE_DIR}/minilm_test_q.npy", test["query_text"].tolist(), "paraphrase-multilingual-MiniLM-L12-v2")

print("Encoding rubert-tiny2 embeddings (cached)...")
rubert_full = get_or_encode(f"{CACHE_DIR}/rubert_full.npy", articles["full_text"].tolist(), "cointegrated/rubert-tiny2")
rubert_title = get_or_encode(f"{CACHE_DIR}/rubert_title.npy", articles["title_text"].tolist(), "cointegrated/rubert-tiny2")
rubert_cal_q = get_or_encode(f"{CACHE_DIR}/rubert_cal_q.npy", calibration["query_text"].tolist(), "cointegrated/rubert-tiny2")
rubert_test_q = get_or_encode(f"{CACHE_DIR}/rubert_test_q.npy", test["query_text"].tolist(), "cointegrated/rubert-tiny2")

def compute_scores(query, query_idx, is_calibration):
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
    
    if is_calibration:
        ml_q = minilm_cal_q[query_idx]
        rb_q = rubert_cal_q[query_idx]
    else:
        ml_q = minilm_test_q[query_idx]
        rb_q = rubert_test_q[query_idx]
    
    scores["emb_minilm"] = minilm_full @ ml_q
    scores["emb_title_minilm"] = minilm_title @ ml_q
    scores["emb_rubert"] = rubert_full @ rb_q
    scores["emb_title_rubert"] = rubert_title @ rb_q
    
    return scores

def build_features(query, query_idx, candidate_ids, is_calibration):
    scores = compute_scores(query, query_idx, is_calibration)
    normalized = {}
    ranks = {}
    for key, s in scores.items():
        smin, smax = s.min(), s.max()
        normalized[key] = (s - smin) / (smax - smin) if smax > smin else np.zeros_like(s)
        ranks[key] = np.argsort(np.argsort(-s))
    
    query_tokens = tokenize(query, False)
    query_tokens_set = set(query_tokens)
    query_bigrams = get_ngrams(query_tokens, 2)
    query_trigrams = get_ngrams(query_tokens, 3)
    query_stems = set(tokenize_stemmed(query, False))
    
    features = []
    for aid in candidate_ids:
        idx = id_to_idx[aid]
        base = []
        for key in sorted(scores.keys()):
            base.extend([scores[key][idx], normalized[key][idx], ranks[key][idx]])
        base.extend([len(query), len(articles.iloc[idx]["full_text"]), len(articles.iloc[idx]["title_text"])])
        
        interactions = [
            base[0] * base[3],
            base[6] * base[18],
            base[18] * base[24],
            base[1] * base[19],
            base[2] * base[20],
            base[6] * base[7],
            base[18] * base[21],
        ]
        
        doc_tokens = articles.iloc[idx]["tokens"]
        title_tokens = articles.iloc[idx]["title_tokens"]
        doc_tokens_set = set(doc_tokens)
        title_tokens_set = set(title_tokens)
        doc_stems = set(articles.iloc[idx]["tokens_stem"])
        
        overlaps = [
            len(query_tokens_set & doc_tokens_set),
            len(query_tokens_set & title_tokens_set),
            jaccard(query_tokens_set, doc_tokens_set),
            jaccard(query_tokens_set, title_tokens_set),
            jaccard(query_stems, doc_stems),
            jaccard(query_bigrams, get_ngrams(doc_tokens, 2)),
            jaccard(query_trigrams, get_ngrams(doc_tokens, 3)),
            jaccard(query_bigrams, get_ngrams(title_tokens, 2)),
            sum((Counter(query_tokens) & Counter(doc_tokens)).values()),
            sum((Counter(query_tokens) & Counter(title_tokens)).values()),
        ]
        
        features.append(base + interactions + overlaps)
    return np.array(features)

def get_candidates(query, query_idx, is_calibration, candidate_k=100):
    scores = compute_scores(query, query_idx, is_calibration)
    rankings = [[article_ids[i] for i in np.argsort(scores[key])[::-1][:candidate_k]] for key in scores.keys()]
    weights = [1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 1.5, 0.8, 0.5, 0.3]
    return rrf(rankings, weights=weights, k=30)[:candidate_k]

print("Building training data...")
all_features = []
all_labels = []
groups = []
for idx, row in calibration.iterrows():
    if idx % 100 == 0:
        print(f"  {idx}/{len(calibration)}")
    query = row["query_text"]
    relevant = set(map(int, row["ground_truth"].split()))
    candidates = get_candidates(query, idx, is_calibration=True, candidate_k=100)
    features = build_features(query, idx, candidates, is_calibration=True)
    labels = [1 if aid in relevant else 0 for aid in candidates]
    all_features.append(features)
    all_labels.extend(labels)
    groups.append(len(candidates))

X_train = np.vstack(all_features)
y_train = np.array(all_labels)
print(f"Training data: {X_train.shape}, positives: {y_train.sum()}")

print("Training LambdaRank...")
train_set = lgb.Dataset(X_train, label=y_train, group=groups)
params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [10],
    "boosting_type": "gbdt",
    "num_leaves": 127,
    "learning_rate": 0.03,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}
ranker = lgb.train(params, train_set, num_boost_round=300)
ranker.save_model("./lgbm_model.txt")

print("Validating on calibration (reference)...")
ap_scores = []
for idx, row in calibration.iterrows():
    query = row["query_text"]
    relevant = list(map(int, row["ground_truth"].split()))
    candidates = get_candidates(query, idx, is_calibration=True, candidate_k=100)
    features = build_features(query, idx, candidates, is_calibration=True)
    preds = ranker.predict(features)
    ranked_indices = np.argsort(preds)[::-1]
    predicted = [candidates[i] for i in ranked_indices[:10]]
    ap_scores.append(map_at_k(predicted, relevant))
print(f"Train MAP@10: {np.mean(ap_scores):.4f}")

print("Generating test predictions...")
results = []
for idx, row in test.iterrows():
    if idx % 100 == 0:
        print(f"  {idx}/{len(test)}")
    query = row["query_text"]
    candidates = get_candidates(query, idx, is_calibration=False, candidate_k=100)
    features = build_features(query, idx, candidates, is_calibration=False)
    preds = ranker.predict(features)
    ranked_indices = np.argsort(preds)[::-1]
    predicted = [candidates[i] for i in ranked_indices[:10]]
    results.append({"query_id": row["query_id"], "answer": " ".join(map(str, predicted))})

pd.DataFrame(results).to_csv("./answer.csv", index=False)
print("Saved answer.csv")

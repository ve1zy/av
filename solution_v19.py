import pandas as pd
import numpy as np
import re
from collections import Counter
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import lightgbm as lgb
from sklearn.model_selection import KFold

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

print("Loading embeddings...")
models = {
    "minilm": SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2"),
    "rubert": SentenceTransformer("cointegrated/rubert-tiny2"),
}

print("Encoding articles...")
art_embs = {}
for name, model in models.items():
    full_emb = model.encode(articles["full_text"].tolist(), batch_size=128, show_progress_bar=False, normalize_embeddings=True)
    title_emb = model.encode(articles["title_text"].tolist(), batch_size=128, show_progress_bar=False, normalize_embeddings=True)
    art_embs[name] = (model, full_emb, title_emb)

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

def get_candidates(query, candidate_k=200):
    scores = get_all_scores(query)
    rankings = [[article_ids[i] for i in np.argsort(scores[key])[::-1][:candidate_k]] for key in scores.keys()]
    weights = [1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 1.5, 0.8, 0.5, 0.3]
    return rrf(rankings, weights=weights, k=30)[:candidate_k]

print("Precomputing features...")
all_queries = calibration["query_text"].tolist()
all_relevant = [set(map(int, x.split())) for x in calibration["ground_truth"].tolist()]

all_candidates = []
all_features = []
all_labels = []
for idx, query in enumerate(all_queries):
    if idx % 100 == 0:
        print(f"  {idx}/{len(all_queries)}")
    candidates = get_candidates(query, candidate_k=200)
    all_candidates.append(candidates)
    features = build_features_for_candidates(query, candidates)
    all_features.append(features)
    labels = [1 if aid in all_relevant[idx] else 0 for aid in candidates]
    all_labels.extend(labels)

X_all = np.vstack(all_features)
y_all = np.array(all_labels)
print(f"Total samples: {X_all.shape[0]}, features: {X_all.shape[1]}, positives: {y_all.sum()}")

print("\n5-Fold CV Ensemble...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
models_list = []

for fold, (train_idx, val_idx) in enumerate(kf.split(all_queries)):
    print(f"\nFold {fold+1}...")
    
    train_starts = [sum(len(all_candidates[j]) for j in range(i)) for i in range(len(all_queries))]
    X_train = np.vstack([all_features[i] for i in train_idx])
    y_train = np.concatenate([all_labels[train_starts[i]:train_starts[i]+len(all_candidates[i])] for i in train_idx])
    train_groups = [len(all_candidates[i]) for i in train_idx]
    
    train_set = lgb.Dataset(X_train, label=y_train, group=train_groups)
    
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
        "seed": fold,
    }
    
    ranker = lgb.train(params, train_set, num_boost_round=300)
    models_list.append(ranker)
    
    ap_scores = []
    for i in val_idx:
        candidates = all_candidates[i]
        features = all_features[i]
        preds = ranker.predict(features)
        ranked_indices = np.argsort(preds)[::-1]
        predicted = [candidates[j] for j in ranked_indices[:10]]
        ap_scores.append(map_at_k(predicted, list(all_relevant[i])))
    
    fold_map = np.mean(ap_scores)
    print(f"Fold {fold+1} single model MAP@10: {fold_map:.4f}")

# Ensemble evaluation
print("\nEnsemble evaluation...")
for fold, (train_idx, val_idx) in enumerate(kf.split(all_queries)):
    ap_scores = []
    for i in val_idx:
        candidates = all_candidates[i]
        features = all_features[i]
        
        # Average predictions from models NOT trained on this validation fold
        preds = np.zeros(len(candidates))
        for m_fold, m in enumerate(models_list):
            if m_fold != fold:
                preds += m.predict(features)
        preds /= 4
        
        ranked_indices = np.argsort(preds)[::-1]
        predicted = [candidates[j] for j in ranked_indices[:10]]
        ap_scores.append(map_at_k(predicted, list(all_relevant[i])))
    
    fold_map = np.mean(ap_scores)
    cv_scores.append(fold_map)
    print(f"Fold {fold+1} ensemble MAP@10: {fold_map:.4f}")

print(f"\nMean Ensemble CV MAP@10: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
print(f"CV scores: {[f'{s:.4f}' for s in cv_scores]}")

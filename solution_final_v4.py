import pandas as pd
import numpy as np
import re
import os
from collections import Counter
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from collections import defaultdict
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import lightgbm as lgb

DATA_DIR = "./candidate_data"
CACHE_DIR = "./cache"
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

# Кэширование base embeddings
base_full_cache = f"{CACHE_DIR}/base_full.npy"
base_title_cache = f"{CACHE_DIR}/base_title.npy"

if os.path.exists(base_full_cache):
    print("Loading cached base embeddings...")
    base_full_emb = np.load(base_full_cache)
    base_title_emb = np.load(base_title_cache)
else:
    print("Encoding base embeddings...")
    base_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    base_full_emb = base_model.encode(articles["full_text"].tolist(), batch_size=128, show_progress_bar=False, normalize_embeddings=True)
    base_title_emb = base_model.encode(articles["title_text"].tolist(), batch_size=128, show_progress_bar=False, normalize_embeddings=True)
    np.save(base_full_cache, base_full_emb)
    np.save(base_title_cache, base_title_emb)

# Fine-tuning MiniLM
ft_full_cache = f"{CACHE_DIR}/ft_full.npy"
ft_title_cache = f"{CACHE_DIR}/ft_title.npy"
ft_query_cache = f"{CACHE_DIR}/ft_queries.npy"
ft_test_query_cache = f"{CACHE_DIR}/ft_test_queries.npy"

if os.path.exists(ft_full_cache):
    print("Loading cached fine-tuned embeddings...")
    ft_full_emb = np.load(ft_full_cache)
    ft_title_emb = np.load(ft_title_cache)
    ft_query_embs = np.load(ft_query_cache)
    ft_test_query_embs = np.load(ft_test_query_cache)
else:
    print("Fine-tuning MiniLM (1 epoch)...")
    
    # Собираем triplets
    train_examples = []
    for _, row in calibration.iterrows():
        query = row["query_text"]
        relevant_ids = list(map(int, row["ground_truth"].split()))
        
        tokens = tokenize(query, False)
        bm25_scores = bm25.get_scores(tokens)
        top_indices = np.argsort(bm25_scores)[::-1][:30]
        top_aids = [article_ids[i] for i in top_indices]
        
        for rel_aid in relevant_ids:
            if rel_aid not in id_to_idx:
                continue
            rel_idx = id_to_idx[rel_aid]
            positive_text = articles.iloc[rel_idx]["full_text"]
            
            negatives = [aid for aid in top_aids if aid not in relevant_ids]
            for neg_aid in negatives[:2]:
                neg_idx = id_to_idx[neg_aid]
                negative_text = articles.iloc[neg_idx]["full_text"]
                train_examples.append(InputExample(texts=[query, positive_text, negative_text]))
    
    print(f"Triplets: {len(train_examples)}")
    
    ft_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.TripletLoss(model=ft_model)
    
    ft_model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        warmup_steps=50,
        show_progress_bar=False,
    )
    
    print("Encoding with fine-tuned model...")
    ft_full_emb = ft_model.encode(articles["full_text"].tolist(), batch_size=128, show_progress_bar=False, normalize_embeddings=True)
    ft_title_emb = ft_model.encode(articles["title_text"].tolist(), batch_size=128, show_progress_bar=False, normalize_embeddings=True)
    
    # Batch encode all queries
    cal_queries = calibration["query_text"].tolist()
    test_queries = test["query_text"].tolist()
    ft_query_embs = ft_model.encode(cal_queries, batch_size=128, show_progress_bar=False, normalize_embeddings=True)
    ft_test_query_embs = ft_model.encode(test_queries, batch_size=128, show_progress_bar=False, normalize_embeddings=True)
    
    np.save(ft_full_cache, ft_full_emb)
    np.save(ft_title_cache, ft_title_emb)
    np.save(ft_query_cache, ft_query_embs)
    np.save(ft_test_query_cache, ft_test_query_embs)

# Batch encode base queries
base_query_cache = f"{CACHE_DIR}/base_queries.npy"
base_test_query_cache = f"{CACHE_DIR}/base_test_queries.npy"

if os.path.exists(base_query_cache):
    base_query_embs = np.load(base_query_cache)
    base_test_query_embs = np.load(base_test_query_cache)
else:
    print("Batch encoding base queries...")
    base_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    cal_queries = calibration["query_text"].tolist()
    test_queries = test["query_text"].tolist()
    base_query_embs = base_model.encode(cal_queries, batch_size=128, show_progress_bar=False, normalize_embeddings=True)
    base_test_query_embs = base_model.encode(test_queries, batch_size=128, show_progress_bar=False, normalize_embeddings=True)
    np.save(base_query_cache, base_query_embs)
    np.save(base_test_query_cache, base_test_query_embs)

# Предвычисляем все scores для calibration и test
print("Precomputing scores...")

def compute_scores_batch(query, query_idx, is_calibration=True):
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
        base_q = base_query_embs[query_idx]
        ft_q = ft_query_embs[query_idx]
    else:
        base_q = base_test_query_embs[query_idx]
        ft_q = ft_test_query_embs[query_idx]
    
    scores["emb_base"] = base_full_emb @ base_q
    scores["emb_title_base"] = base_title_emb @ base_q
    scores["emb_ft"] = ft_full_emb @ ft_q
    scores["emb_title_ft"] = ft_title_emb @ ft_q
    
    return scores

def build_features(query, query_idx, candidate_ids, is_calibration=True):
    scores = compute_scores_batch(query, query_idx, is_calibration)
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
        
        n_score = len(sorted(scores.keys()))
        interactions = []
        for i in range(n_score):
            for j in range(i+1, n_score):
                interactions.append(base[i*3] * base[j*3])
                interactions.append(base[i*3+1] * base[j*3+1])
        
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

def get_candidates(query, query_idx, is_calibration=True, candidate_k=200):
    scores = compute_scores_batch(query, query_idx, is_calibration)
    rankings = [[article_ids[i] for i in np.argsort(scores[key])[::-1][:candidate_k]] for key in scores.keys()]
    weights = [1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 1.5, 0.8, 1.5, 0.5, 1.5, 0.5]
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
    candidates = get_candidates(query, idx, is_calibration=True, candidate_k=200)
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
    "num_leaves": 31,
    "learning_rate": 0.02,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.6,
    "bagging_freq": 5,
    "min_data_in_leaf": 20,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
    "verbose": -1,
}
ranker = lgb.train(params, train_set, num_boost_round=150)
ranker.save_model("./lgbm_model.txt")

print("Validating on calibration (reference)...")
ap_scores = []
for idx, row in calibration.iterrows():
    query = row["query_text"]
    relevant = list(map(int, row["ground_truth"].split()))
    candidates = get_candidates(query, idx, is_calibration=True, candidate_k=200)
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
    candidates = get_candidates(query, idx, is_calibration=False, candidate_k=200)
    features = build_features(query, idx, candidates, is_calibration=False)
    preds = ranker.predict(features)
    ranked_indices = np.argsort(preds)[::-1]
    predicted = [candidates[i] for i in ranked_indices[:10]]
    results.append({"query_id": row["query_id"], "answer": " ".join(map(str, predicted))})

pd.DataFrame(results).to_csv("./answer.csv", index=False)
print("Saved answer.csv")

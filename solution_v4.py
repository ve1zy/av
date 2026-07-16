import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from nltk.stem import SnowballStemmer
import lightgbm as lgb

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
    tokens = text.split()
    return tokens

def tokenize_with_stemming(text):
    text = text.lower()
    text = re.sub(r"[^a-zа-яё0-9\s]", " ", text)
    tokens = text.split()
    stemmed = [stemmer.stem(t) for t in tokens]
    return stemmed

def map_at_k(predicted, relevant, k=10):
    predicted = predicted[:k]
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    score = 0.0
    hits = 0
    for i, pid in enumerate(predicted):
        if pid in relevant_set:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(relevant_set), k)

def reciprocal_rank_fusion(rankings, weights=None, k=60):
    if weights is None:
        weights = [1.0] * len(rankings)
    scores = defaultdict(float)
    for ranking, weight in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] += weight / (k + rank + 1)
    sorted_docs = sorted(scores.items(), key=lambda x: -x[1])
    return [doc_id for doc_id, _ in sorted_docs]

print("Loading data...")
articles = pd.read_feather(f"{DATA_DIR}/articles.f")
calibration = pd.read_feather(f"{DATA_DIR}/calibration.f")
test = pd.read_feather(f"{DATA_DIR}/test.f")

print("Cleaning HTML...")
articles["clean_body"] = articles["body"].apply(clean_html)
articles["title_weighted"] = articles["title"].apply(lambda x: (x + " ") * 3)
articles["full_text"] = articles["title_weighted"] + articles["clean_body"]
articles["tokens"] = articles["full_text"].apply(tokenize)
articles["tokens_stemmed"] = articles["full_text"].apply(tokenize_with_stemming)

corpus_tokens = articles["tokens"].tolist()
corpus_tokens_stemmed = articles["tokens_stemmed"].tolist()
article_ids = articles["article_id"].tolist()
id_to_idx = {aid: i for i, aid in enumerate(article_ids)}

print("Building BM25 indices...")
bm25 = BM25Okapi(corpus_tokens)
bm25_stemmed = BM25Okapi(corpus_tokens_stemmed)

print("Building TF-IDF index...")
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(articles["full_text"].tolist())

print("Loading sentence transformer model...")
bi_encoder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

print("Encoding articles...")
article_texts = articles["full_text"].tolist()
article_embeddings = bi_encoder.encode(article_texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)

def get_bm25_score(query, article_idx, stemmed=False):
    if stemmed:
        tokens = tokenize_with_stemming(query)
        scores = bm25_stemmed.get_scores(tokens)
    else:
        tokens = tokenize(query)
        scores = bm25.get_scores(tokens)
    return scores[article_idx]

def get_tfidf_score(query, article_idx):
    query_vec = tfidf.transform([query])
    scores = (tfidf_matrix @ query_vec.T).toarray().flatten()
    return scores[article_idx]

def get_embedding_score(query, article_idx):
    query_emb = bi_encoder.encode([query], normalize_embeddings=True)[0]
    return np.dot(article_embeddings[article_idx], query_emb)

def get_candidate_scores(query, candidate_ids):
    features = []
    for aid in candidate_ids:
        idx = id_to_idx[aid]
        bm25_s = get_bm25_score(query, idx, stemmed=False)
        bm25_stem_s = get_bm25_score(query, idx, stemmed=True)
        tfidf_s = get_tfidf_score(query, idx)
        emb_s = get_embedding_score(query, idx)
        features.append([bm25_s, bm25_stem_s, tfidf_s, emb_s])
    return np.array(features)

def get_hybrid_ranking(query, top_k=50):
    tokens = tokenize(query)
    bm25_scores = bm25.get_scores(tokens)
    tokens_stemmed = tokenize_with_stemming(query)
    bm25_stem_scores = bm25_stemmed.get_scores(tokens_stemmed)
    query_vec = tfidf.transform([query])
    tfidf_scores = (tfidf_matrix @ query_vec.T).toarray().flatten()
    query_emb = bi_encoder.encode([query], normalize_embeddings=True)[0]
    emb_scores = np.dot(article_embeddings, query_emb)
    
    bm25_ranking = [article_ids[i] for i in np.argsort(bm25_scores)[::-1][:50]]
    bm25_stem_ranking = [article_ids[i] for i in np.argsort(bm25_stem_scores)[::-1][:50]]
    tfidf_ranking = [article_ids[i] for i in np.argsort(tfidf_scores)[::-1][:50]]
    emb_ranking = [article_ids[i] for i in np.argsort(emb_scores)[::-1][:50]]
    
    fused = reciprocal_rank_fusion(
        [bm25_ranking, bm25_stem_ranking, tfidf_ranking, emb_ranking],
        weights=[1.0, 0.8, 1.0, 1.5],
        k=60
    )
    return fused[:top_k]

print("Building training data from calibration...")
train_data = []
for idx, row in calibration.iterrows():
    query = row["query_text"]
    relevant = set(map(int, row["ground_truth"].split()))
    
    candidates = get_hybrid_ranking(query, top_k=50)
    features = get_candidate_scores(query, candidates)
    
    for i, aid in enumerate(candidates):
        label = 1 if aid in relevant else 0
        train_data.append([label] + features[i].tolist())

train_df = pd.DataFrame(train_data, columns=["label", "bm25", "bm25_stem", "tfidf", "emb"])
print(f"Training data shape: {train_df.shape}")
print(f"Positive samples: {train_df['label'].sum()}")

print("Training LightGBM ranker...")
X_train = train_df[["bm25", "bm25_stem", "tfidf", "emb"]].values
y_train = train_df["label"].values

train_set = lgb.Dataset(X_train, label=y_train)
params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "verbose": -1
}

model = lgb.train(params, train_set, num_boost_round=100)

def get_lgbm_ranking(query, top_k=10):
    candidates = get_hybrid_ranking(query, top_k=50)
    features = get_candidate_scores(query, candidates)
    preds = model.predict(features)
    ranked_indices = np.argsort(preds)[::-1]
    return [candidates[i] for i in ranked_indices[:top_k]]

print("Validating on calibration set with LightGBM...")
ap_scores = []
for idx, row in calibration.iterrows():
    if idx % 100 == 0:
        print(f"  Processing {idx}/{len(calibration)}...")
    query = row["query_text"]
    relevant = list(map(int, row["ground_truth"].split()))
    predicted = get_lgbm_ranking(query, top_k=10)
    ap = map_at_k(predicted, relevant, k=10)
    ap_scores.append(ap)

map_score = np.mean(ap_scores)
print(f"MAP@10 on calibration (LightGBM): {map_score:.4f}")

print("Generating predictions for test set...")
results = []
for idx, row in test.iterrows():
    if idx % 100 == 0:
        print(f"  Processing {idx}/{len(test)}...")
    query_id = row["query_id"]
    query = row["query_text"]
    predicted = get_lgbm_ranking(query, top_k=10)
    answer = " ".join(map(str, predicted))
    results.append({"query_id": query_id, "answer": answer})

answer_df = pd.DataFrame(results)
answer_df.to_csv("C:/Users/PC/Downloads/avito/answer.csv", index=False)
print("Saved answer.csv")
print(f"Total test queries: {len(answer_df)}")

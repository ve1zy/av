import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi
from scipy.spatial.distance import cosine
from collections import defaultdict

DATA_DIR = "C:/Users/PC/Downloads/avito/candidate_data"

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

def reciprocal_rank_fusion(rankings, k=60):
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] += 1.0 / (k + rank + 1)
    sorted_docs = sorted(scores.items(), key=lambda x: -x[1])
    return [doc_id for doc_id, _ in sorted_docs]

print("Loading data...")
articles = pd.read_feather(f"{DATA_DIR}/articles.f")
calibration = pd.read_feather(f"{DATA_DIR}/calibration.f")
test = pd.read_feather(f"{DATA_DIR}/test.f")

print("Cleaning HTML...")
articles["clean_body"] = articles["body"].apply(clean_html)
articles["full_text"] = articles["title"] + " " + articles["clean_body"]
articles["tokens"] = articles["full_text"].apply(tokenize)

corpus_tokens = articles["tokens"].tolist()
article_ids = articles["article_id"].tolist()
id_to_idx = {aid: i for i, aid in enumerate(article_ids)}

print("Building BM25 index...")
bm25 = BM25Okapi(corpus_tokens)

print("Loading sentence transformer model...")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

print("Encoding articles...")
article_texts = articles["full_text"].tolist()
article_embeddings = model.encode(article_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

def get_bm25_ranking(query, top_k=50):
    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [article_ids[i] for i in top_indices]

def get_embedding_ranking(query, top_k=50):
    query_emb = model.encode([query], normalize_embeddings=True)[0]
    similarities = np.dot(article_embeddings, query_emb)
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [article_ids[i] for i in top_indices]

def get_hybrid_ranking(query, top_k=10):
    bm25_ranking = get_bm25_ranking(query, top_k=50)
    emb_ranking = get_embedding_ranking(query, top_k=50)
    fused = reciprocal_rank_fusion([bm25_ranking, emb_ranking], k=60)
    return fused[:top_k]

print("Validating on calibration set...")
ap_scores = []
for _, row in calibration.iterrows():
    query = row["query_text"]
    relevant = list(map(int, row["ground_truth"].split()))
    predicted = get_hybrid_ranking(query, top_k=10)
    ap = map_at_k(predicted, relevant, k=10)
    ap_scores.append(ap)

map_score = np.mean(ap_scores)
print(f"MAP@10 on calibration: {map_score:.4f}")

print("Generating predictions for test set...")
results = []
for _, row in test.iterrows():
    query_id = row["query_id"]
    query = row["query_text"]
    predicted = get_hybrid_ranking(query, top_k=10)
    answer = " ".join(map(str, predicted))
    results.append({"query_id": query_id, "answer": answer})

answer_df = pd.DataFrame(results)
answer_df.to_csv("C:/Users/PC/Downloads/avito/answer.csv", index=False)
print("Saved answer.csv")
print(f"Total test queries: {len(answer_df)}")

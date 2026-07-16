import pandas as pd
import numpy as np
import re
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

print("Loading base embeddings...")
base_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
base_full_emb = base_model.encode(articles["full_text"].tolist(), batch_size=128, show_progress_bar=False, normalize_embeddings=True)
base_title_emb = base_model.encode(articles["title_text"].tolist(), batch_size=128, show_progress_bar=False, normalize_embeddings=True)

print("Loading other embeddings...")
models = {
    "rubert": SentenceTransformer("cointegrated/rubert-tiny2"),
    "e5": SentenceTransformer("intfloat/multilingual-e5-base"),
}

art_embs = {}
for name, model in models.items():
    print(f"  {name}...")
    full_emb = model.encode(articles["full_text"].tolist(), batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    title_emb = model.encode(articles["title_text"].tolist(), batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    art_embs[name] = (model, full_emb, title_emb)

print("\nPreparing triplets for fine-tuning...")
# 5-fold CV: для каждого фолда fine-tune модель на 4 фолдах и оцениваем на 5-м
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(calibration)):
    print(f"\n{'='*50}")
    print(f"Fold {fold+1}")
    print(f"{'='*50}")
    
    train_cal = calibration.iloc[train_idx].reset_index(drop=True)
    val_cal = calibration.iloc[val_idx].reset_index(drop=True)
    
    # Собираем triplets: anchor=query, positive=ground truth doc, negative=hard negative из BM25 top-20
    train_examples = []
    for _, row in train_cal.iterrows():
        query = row["query_text"]
        relevant_ids = list(map(int, row["ground_truth"].split()))
        
        tokens = tokenize(query, False)
        bm25_scores = bm25.get_scores(tokens)
        top_indices = np.argsort(bm25_scores)[::-1][:30]
        top_aids = [article_ids[i] for i in top_indices]
        
        for rel_aid in relevant_ids:
            rel_idx = id_to_idx[rel_aid]
            positive_text = articles.iloc[rel_idx]["full_text"]
            
            # Hard negative: высокий BM25 score, но не релевантный
            negatives = [aid for aid in top_aids if aid not in relevant_ids]
            for neg_aid in negatives[:3]:
                neg_idx = id_to_idx[neg_aid]
                negative_text = articles.iloc[neg_idx]["full_text"]
                train_examples.append(InputExample(texts=[query, positive_text, negative_text]))
    
    print(f"Train triplets: {len(train_examples)}")
    
    # Fine-tune модель
    print("Fine-tuning MiniLM...")
    ft_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.TripletLoss(model=ft_model)
    
    ft_model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=2,
        warmup_steps=100,
        show_progress_bar=False,
    )
    
    # Кодируем статьи fine-tuned моделью
    print("Encoding articles with fine-tuned model...")
    ft_full_emb = ft_model.encode(articles["full_text"].tolist(), batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    ft_title_emb = ft_model.encode(articles["title_text"].tolist(), batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    
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
            "emb_minilm_full": base_full_emb @ base_model.encode([query], show_progress_bar=False, normalize_embeddings=True)[0],
            "emb_minilm_title": base_title_emb @ base_model.encode([query], show_progress_bar=False, normalize_embeddings=True)[0],
            "emb_ft_full": ft_full_emb @ ft_model.encode([query], show_progress_bar=False, normalize_embeddings=True)[0],
            "emb_ft_title": ft_title_emb @ ft_model.encode([query], show_progress_bar=False, normalize_embeddings=True)[0],
        }
        
        for name, (model, full_emb, title_emb) in art_embs.items():
            q_emb = model.encode([query], show_progress_bar=False, normalize_embeddings=True)[0]
            scores[f"emb_{name}"] = full_emb @ q_emb
            scores[f"emb_title_{name}"] = title_emb @ q_emb
        
        return scores
    
    def build_features(query, candidate_ids):
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
                sum((Counter(query_tokens) & Counter(doc_tokens)).values()),
            ]
            
            features.append(base + interactions + overlaps)
        return np.array(features)
    
    def get_candidates(query, candidate_k=200):
        scores = get_all_scores(query)
        rankings = [[article_ids[i] for i in np.argsort(scores[key])[::-1][:candidate_k]] for key in scores.keys()]
        weights = [1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 1.5, 0.8, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5]
        return rrf(rankings, weights=weights, k=30)[:candidate_k]
    
    # Build training data for LGBM
    print("Building LGBM features...")
    train_features = []
    train_labels = []
    train_groups = []
    for _, row in train_cal.iterrows():
        query = row["query_text"]
        relevant = set(map(int, row["ground_truth"].split()))
        candidates = get_candidates(query, candidate_k=200)
        features = build_features(query, candidates)
        labels = [1 if aid in relevant else 0 for aid in candidates]
        train_features.append(features)
        train_labels.extend(labels)
        train_groups.append(len(candidates))
    
    X_train = np.vstack(train_features)
    y_train = np.array(train_labels)
    
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
    }
    
    ranker = lgb.train(params, train_set, num_boost_round=300)
    
    # Validation
    print("Validating...")
    ap_scores = []
    for _, row in val_cal.iterrows():
        query = row["query_text"]
        relevant = list(map(int, row["ground_truth"].split()))
        candidates = get_candidates(query, candidate_k=200)
        features = build_features(query, candidates)
        preds = ranker.predict(features)
        ranked_indices = np.argsort(preds)[::-1]
        predicted = [candidates[j] for j in ranked_indices[:10]]
        ap_scores.append(map_at_k(predicted, relevant))
    
    fold_map = np.mean(ap_scores)
    cv_scores.append(fold_map)
    print(f"Fold {fold+1} MAP@10: {fold_map:.4f}")

print(f"\nMean CV MAP@10: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
print(f"CV scores: {[f'{s:.4f}' for s in cv_scores]}")

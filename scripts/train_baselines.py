"""
Leakage-Free train_baselines.py
-------------------------------
Baselines implemented:

1. Popularity
2. Classic CBF (item-item cosine similarity on tags)
3. Item-CF (pure rating-based item cosine)
4. User-CF (mean-centered rating vectors, classic CF)
5. SVD Hybrid (Matrix Factorization + PCA item embeddings)

All learning stages use TRAIN ONLY.
df_dense is used ONLY to build item index.
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import random
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA


# ============================================================
# Reproducibility
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# For CUDA (harmless on CPU-only systems)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# ============================================================
# Paths
# ============================================================
DATA = "new_data"
MODELS = "models"
os.makedirs(MODELS, exist_ok=True)

TRAIN = f"{DATA}/train_df.pkl"
DENSE = f"{DATA}/df_dense.pkl"       # used ONLY for item index (safe)
EVAL = f"{DATA}/eval_data.pkl"
BOOK_GENRES = f"{DATA}/book_genres_filtered.csv"

# ============================================================
# Metrics
# ============================================================
def HR5(ranked, gt):
    return 1.0 if gt in ranked[:5] else 0.0

def PREC5(ranked, gt):
    return 1/5 if gt in ranked[:5] else 0.0

def NDCG5(ranked, gt):
    if gt in ranked[:5]:
        i = ranked.index(gt)
        return 1 / np.log2(i + 2)
    return 0.0

def evaluate(name, scorer, eval_data):
    hr = ndcg = prec = 0
    total = len(eval_data)
    for u, info in eval_data.items():
        gt = info["gt"]
        candidates = [gt] + info["neg_items"]

        scores = scorer(u, candidates)
        ranked = sorted(candidates, key=lambda b: scores[b], reverse=True)

        hr += HR5(ranked, gt)
        ndcg += NDCG5(ranked, gt)
        prec += PREC5(ranked, gt)

    return {"model": name, "HR@5": hr/total, "NDCG@5": ndcg/total, "P@5": prec/total}

# ============================================================
# Load Data
# ============================================================
print("\n===== LOADING DATA =====")
train_df = pd.read_pickle(TRAIN)
df_dense = pd.read_pickle(DENSE)   # ONLY for item index
with open(EVAL, "rb") as f: eval_data = pickle.load(f)
book_genres = pd.read_csv(BOOK_GENRES)

# ============================================================
# Item index (from df_dense only)
# ============================================================
print("Building item index...")
all_books = sorted(df_dense.book_id.unique())
book_to_idx = {b: i for i, b in enumerate(all_books)}
idx_to_book = {i: b for b, i in book_to_idx.items()}
num_items = len(all_books)

# Filter genres
book_genres = book_genres[book_genres.book_id.isin(all_books)].reset_index(drop=True)

# ============================================================
# Build genre matrix
# ============================================================
print("Building genre embeddings...")

genre_list = sorted({
    g.strip()
    for row in book_genres.genres
    for g in row.split(",")
    if g.strip()
})
genre_to_idx = {g: i for i, g in enumerate(genre_list)}
num_genres = len(genre_list)

tag_matrix = np.zeros((num_items, num_genres), dtype=np.float32)

for _, row in book_genres.iterrows():
    b = row.book_id
    if b not in book_to_idx: continue
    bi = book_to_idx[b]
    for g in row.genres.split(","):
        g = g.strip()
        if g in genre_to_idx:
            tag_matrix[bi, genre_to_idx[g]] = 1.0

tag_norm = tag_matrix / (np.sqrt(tag_matrix.sum(axis=1, keepdims=True)) + 1e-8)

# ============================================================
# Classic CBF similarity (item–item cosine)
# ============================================================
print("Computing CBF similarity...")
cbf_sim = cosine_similarity(tag_norm)

def cbf_scorer(user, candidates):
    rated = user_items.get(user, [])
    if len(rated) == 0:
        return {b: 0 for b in candidates}

    rated_idx = [book_to_idx[b] for b in rated]
    out = {}
    for b in candidates:
        bi = book_to_idx[b]
        out[b] = cbf_sim[bi, rated_idx].mean()
    return out

# ============================================================
# Popularity baseline
# ============================================================
print("Training Popularity baseline...")
pop_counts = train_df.book_id.value_counts()

def popularity_scorer(user, candidates):
    return {b: pop_counts.get(b, 0) for b in candidates}

pickle.dump(pop_counts, open(f"{MODELS}/popularity.pkl", "wb"))

# ============================================================
# User Histories (TRAIN ONLY)
# ============================================================
print("Building user histories...")
user_ids = sorted(train_df.user_id.unique())
user_to_idx = {u: i for i, u in enumerate(user_ids)}

user_items = train_df.groupby("user_id")["book_id"].apply(list).to_dict()

# ============================================================
# Item-CF: Pure rating-based cosine
# ============================================================
print("Building Item-CF similarity...")

num_users = len(user_ids)
R = np.zeros((num_users, num_items))

# Fill TRAIN ONLY ratings
for _, row in train_df.iterrows():
    R[user_to_idx[row.user_id], book_to_idx[row.book_id]] = row.rating

item_sim = cosine_similarity(R.T)  # item-item cosine

def item_cf_scorer(user, candidates):
    rated = user_items.get(user, [])
    if len(rated) == 0: return {b: 0 for b in candidates}

    rated_idx = [book_to_idx[b] for b in rated]
    out = {}
    for b in candidates:
        bi = book_to_idx[b]
        out[b] = item_sim[bi, rated_idx].mean()
    return out

pickle.dump(item_sim, open(f"{MODELS}/item_cf.pkl", "wb"))

# ============================================================
# USER-CF: Mean-centered rating CF (classic)
# ============================================================
print("Building User-CF (mean-centered rating vectors)...")

# Build user rating dict
user_ratings = (
    train_df.groupby("user_id")[["book_id","rating"]]
    .apply(lambda df: dict(zip(df.book_id, df.rating)))
    .to_dict()
)

# Build mean-centered rating vectors
user_rating_vectors = np.zeros((num_users, num_items))

for u in user_ids:
    uid = user_to_idx[u]
    items_rated = user_ratings.get(u, {})
    if len(items_rated) == 0: continue

    mean_r = np.mean(list(items_rated.values()))
    for b, r in items_rated.items():
        user_rating_vectors[uid, book_to_idx[b]] = r - mean_r

user_sim = cosine_similarity(user_rating_vectors)  # true user-user CF

def user_cf_scorer(user, candidates):
    uid = user_to_idx[user]
    sim_u = user_sim[uid]

    rated_users = {}
    out = {}

    for b in candidates:
        raters = train_df[train_df.book_id == b].user_id.values
        if len(raters) == 0:
            out[b] = 0
        else:
            out[b] = np.mean([sim_u[user_to_idx[r]] for r in raters])
    return out

pickle.dump(user_sim, open(f"{MODELS}/user_cf.pkl", "wb"))

# ============================================================
# SVD Hybrid (MF + PCA)
# ============================================================
print("Training Hybrid SVD Early Fusion...")

pca = PCA(n_components=50)
tag_pca_items = pca.fit_transform(tag_norm)

# User tag PCA = mean of items user rated
tag_pca_users = np.zeros((num_users, 50))
for u in user_ids:
    uid = user_to_idx[u]
    books = user_items.get(u, [])
    if books:
        tag_pca_users[uid] = pca.transform(
            tag_norm[[book_to_idx[b] for b in books]].mean(axis=0).reshape(1, -1)
        )

latent_dim = 50
U = np.random.normal(0, 0.05, (num_users, latent_dim))
V = np.random.normal(0, 0.05, (num_items, latent_dim))

triples = [
    (user_to_idx[u], book_to_idx[b], r)
    for u, b, r in zip(train_df.user_id, train_df.book_id, train_df.rating)
]

print("Running MF training (8 epochs)...")
lr = 0.01
reg = 0.05

for ep in range(8):
    np.random.shuffle(triples)
    for uid, bid, r in triples:
        pred = U[uid].dot(V[bid])
        err = r - pred
        U[uid] += lr * (err * V[bid] - reg * U[uid])
        V[bid] += lr * (err * U[uid] - reg * V[bid])

hybrid_user_embed = np.concatenate([U, tag_pca_users], axis=1)
hybrid_item_embed = np.concatenate([V, tag_pca_items], axis=1)

def svd_scorer(user, candidates):
    uid = user_to_idx[user]
    u_vec = hybrid_user_embed[uid]
    return {b: u_vec.dot(hybrid_item_embed[book_to_idx[b]]) for b in candidates}

pickle.dump(
    {"user_embed": hybrid_user_embed,
     "item_embed": hybrid_item_embed,
     "user_to_idx": user_to_idx,
     "book_to_idx": book_to_idx},
    open(f"{MODELS}/svd_hybrid.pkl", "wb")
)

# ============================================================
# Evaluate all baselines
# ============================================================
print("\n===== Evaluating Models =====")
results = []
results.append(evaluate("Popularity", popularity_scorer, eval_data))
results.append(evaluate("CBF (Classic Item-Item)", cbf_scorer, eval_data))
results.append(evaluate("Item-CF (Pure Rating)", item_cf_scorer, eval_data))
results.append(evaluate("User-CF (Mean-Centered)", user_cf_scorer, eval_data))
results.append(evaluate("SVD Hybrid", svd_scorer, eval_data))

df_out = pd.DataFrame(results)
df_out.to_csv(f"{MODELS}/baseline_results.csv", index=False)

print("\n===== DONE =====")
print(df_out)

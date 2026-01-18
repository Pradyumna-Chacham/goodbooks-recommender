"""
inference.py — Multi-Model Recommender Comparison
-------------------------------------------------

Models used:

- User-CF: user–user similarity (models/user_cf.pkl)
- Item-CF: item–item similarity (models/item_cf.pkl)
- CBF: TF-IDF over (title + authors) cosine similarity
- RL: QNet over [user_embed || cf_vec] features (models/rl_cfhard_fast.pth)
- SVD: cosine similarity in latent space (models/svd_hybrid.pkl)
- Hybrid: Z-score blending of Item-CF + RL
          score = CF_WEIGHT * z(CF) + (1 - CF_WEIGHT) * z(RL)
          (CF_WEIGHT loaded from models/hybrid_reranker.pkl)

Data:

- new_data/train_df.pkl          (user_id, book_id, rating)
- data/book_metadata.csv         (book_id, title, authors)

Interactive modes:

1. Random user → show recommendations from ALL models
2. Enter book title → fuzzy match → show Item-CF / SVD / RL / Hybrid / CBF similar books
3. Enter user ID → show recommendations from ALL models
4. Exit
"""

import os
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rapidfuzz import process, fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PATHS
# ============================================================

MODELS = "models"
DATA = "new_data"
RAW = "data"

SVD_HYBRID_PATH = f"{MODELS}/svd_hybrid.pkl"
ITEM_CF_PATH = f"{MODELS}/item_cf.pkl"
USER_CF_PATH = f"{MODELS}/user_cf.pkl"
RL_WEIGHTS_PATH = f"{MODELS}/rl_cfhard_fast.pth"
HYBRID_RERANKER_PATH = f"{MODELS}/hybrid_reranker.pkl"

TRAIN_DF_PATH = f"{DATA}/train_df.pkl"
BOOK_META_CSV = f"{RAW}/book_metadata.csv"


# ============================================================
# RL MODEL (same as training)
# ============================================================

class QNet(nn.Module):
    def __init__(self, dim=150):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# LOADING UTILITIES
# ============================================================

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_artifacts():
    print("⏳ Loading inference artifacts...")

    # -------------------------------
    # Load SVD hybrid embeddings
    # -------------------------------
    svd = load_pickle(SVD_HYBRID_PATH)
    user_embed = torch.tensor(svd["user_embed"], dtype=torch.float32)
    item_embed = torch.tensor(svd["item_embed"], dtype=torch.float32)
    user_to_idx = svd["user_to_idx"]
    book_to_idx = svd["book_to_idx"]
    idx_to_book = {v: k for k, v in book_to_idx.items()}
    idx_to_user = {v: k for k, v in user_to_idx.items()}

    # Also keep numpy copies for cosine SVD
    user_embed_np = user_embed.numpy()
    item_embed_np = item_embed.numpy()
    item_norms = np.linalg.norm(item_embed_np, axis=1) + 1e-9

    # -------------------------------
    # Load Item-CF and User-CF matrices
    # -------------------------------
    item_sim = load_pickle(ITEM_CF_PATH)      # [num_items, num_items]
    user_sim = load_pickle(USER_CF_PATH)      # [num_users, num_users]

    # -------------------------------
    # Load book metadata
    # -------------------------------
    meta_df = pd.read_csv(BOOK_META_CSV)
    meta_df["book_id"] = meta_df["book_id"].astype(int)
    book_meta = {
        int(row.book_id): {
            "title": str(row.title),
            "authors": str(row.authors),
        }
        for _, row in meta_df.iterrows()
    }

    # -------------------------------
    # Build Content-Based TF-IDF features (Option A)
    # -------------------------------
    print("⏳ Building TF-IDF matrix for Content-Based Filtering...")
    cbf_book_ids = list(meta_df["book_id"].values)
    cbf_texts = [
        f"{book_meta[b]['title']} {book_meta[b]['authors']}"
        for b in cbf_book_ids
    ]
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    cbf_content = tfidf.fit_transform(cbf_texts)  # sparse [num_books, num_terms]
    cbf_book_to_row = {bid: i for i, bid in enumerate(cbf_book_ids)}

    # -------------------------------
    # Load training DF for user history
    # -------------------------------
    train_df = pd.read_pickle(TRAIN_DF_PATH)
    user_ratings = (
        train_df.groupby("user_id")[["book_id", "rating"]]
        .apply(lambda df: dict(zip(df["book_id"], df["rating"])))
        .to_dict()
    )
    user_hist = (
        train_df.groupby("user_id")["book_id"]
        .apply(list)
        .to_dict()
    )

    # -------------------------------
    # Precompute CF vectors (top-50 neighbors)
    # -------------------------------
    print("⏳ Precomputing CF vectors (top-50 item neighbors)...")
    num_items = len(book_to_idx)
    TOPK = 50
    cf_vecs_np = np.zeros((num_items, 50), dtype=np.float32)

    for i in range(num_items):
        sims = item_sim[i]
        top_idx = np.argsort(sims)[-TOPK:]
        top_emb = item_embed[top_idx]
        cf_vecs_np[i] = top_emb.mean(dim=0)[:50].numpy()

    cf_vecs = torch.tensor(cf_vecs_np, dtype=torch.float32)

    # -------------------------------
    # Load RL model
    # -------------------------------
    print("⏳ Loading RL Q-network weights...")
    device = torch.device("cpu")
    qnet = QNet(dim=150).to(device)
    qnet.load_state_dict(torch.load(RL_WEIGHTS_PATH, map_location=device, weights_only=True))
    qnet.eval()

    # -------------------------------
    # Load Hybrid reranker config
    # -------------------------------
    hybrid = load_pickle(HYBRID_RERANKER_PATH)
    CF_WEIGHT = hybrid["CF_WEIGHT"]

    print("✅ All artifacts loaded successfully!\n")

    return {
        "device": device,
        "user_embed_t": user_embed.to(device),
        "item_embed_t": item_embed.to(device),
        "cf_vecs": cf_vecs.to(device),
        "user_embed_np": user_embed_np,
        "item_embed_np": item_embed_np,
        "item_norms": item_norms,
        "item_sim": item_sim,
        "user_sim": user_sim,
        "user_to_idx": user_to_idx,
        "book_to_idx": book_to_idx,
        "idx_to_book": idx_to_book,
        "idx_to_user": idx_to_user,
        "book_meta": book_meta,
        "user_ratings": user_ratings,
        "user_hist": user_hist,
        "qnet": qnet,
        "CF_WEIGHT": CF_WEIGHT,
        # Content-based artifacts
        "cbf_book_ids": cbf_book_ids,
        "cbf_book_to_row": cbf_book_to_row,
        "cbf_content": cbf_content,
        "cbf_vectorizer": tfidf,
    }


# ============================================================
# LOW-LEVEL SCORING HELPERS
# ============================================================

def zscore_dict(d):
    if not d:
        return {}
    vals = np.array(list(d.values()), dtype=np.float32)
    mean = vals.mean()
    std = vals.std() + 1e-9
    return {k: (v - mean) / std for k, v in d.items()}


def zscore_array(x):
    x = np.asarray(x, dtype=np.float32)
    return (x - x.mean()) / (x.std() + 1e-9)


# ---------- RL scoring (user + item) ----------

def score_with_rl(user_id, candidates, A):
    """
    Pure RL scoring:
        state = [user_embed(100) || zeros(50)]
        item  = [item_embed(100) || cf_vec(50)]
        input = state * item  (elementwise)
    """
    qnet = A["qnet"]
    device = A["device"]

    user_to_idx = A["user_to_idx"]
    book_to_idx = A["book_to_idx"]

    user_embed = A["user_embed_t"]
    item_embed = A["item_embed_t"]
    cf_vecs = A["cf_vecs"]

    if user_id not in user_to_idx:
        return np.zeros(len(candidates))

    uid = user_to_idx[user_id]
    state_vec = torch.cat([user_embed[uid], torch.zeros(50, device=device)])  # (150,)

    item_vecs = []
    valid_idx = []

    for i, b in enumerate(candidates):
        bi = book_to_idx.get(b)
        if bi is None:
            continue
        item_vec = item_embed[bi]
        cf_vec = cf_vecs[bi]
        full_vec = torch.cat([item_vec, cf_vec])  # (150,)
        item_vecs.append(full_vec)
        valid_idx.append(i)

    if not item_vecs:
        return np.zeros(len(candidates))

    V = torch.stack(item_vecs)                              # (N, 150)
    S = state_vec.unsqueeze(0).expand(len(V), -1)           # (N, 150)
    SA = S * V                                              # (N, 150)

    with torch.no_grad():
        q_vals = qnet(SA).squeeze(1)                        # (N,)

    out = np.zeros(len(candidates), dtype=np.float32)
    q_np = q_vals.cpu().numpy()
    for k, idx in enumerate(valid_idx):
        out[idx] = q_np[k]
    return out


# ============================================================
# USER-LEVEL MODELS
# ============================================================

# ---------- Item-CF scoring for user ----------

def score_candidates_with_itemcf(history, candidates, A):
    """
    Item-CF scoring for a user:
    each candidate is scored by mean similarity to the user's history.
    """
    book_to_idx = A["book_to_idx"]
    item_sim = A["item_sim"]

    hist_idx = [book_to_idx[b] for b in history if b in book_to_idx]
    if not hist_idx:
        return {c: 0.0 for c in candidates}

    scores = {}
    for c in candidates:
        ci = book_to_idx.get(c)
        if ci is None:
            scores[c] = 0.0
        else:
            sims = [item_sim[ci, hi] for hi in hist_idx]
            scores[c] = float(np.mean(sims))
    return scores


# ---------- User-CF scoring ----------

def recommend_usercf_for_user(user_id, top_k, A):
    """
    User-CF recommendations:
    - find similar users via user_sim
    - aggregate their ratings (weighted by similarity)
    """
    user_to_idx = A["user_to_idx"]
    idx_to_user = A["idx_to_user"]
    user_sim = A["user_sim"]
    user_ratings = A["user_ratings"]

    if user_id not in user_to_idx:
        return []

    uid = user_to_idx[user_id]
    sims = user_sim[uid].copy()               # (num_users,)

    # Exclude self
    sims[uid] = -1.0
    top_neighbors = np.argsort(sims)[-50:]    # top-50 neighbors

    target_ratings = user_ratings.get(user_id, {})
    rated_books = set(target_ratings.keys())

    scores_sum = {}
    weights_sum = {}

    for nb_idx in top_neighbors:
        nb_user = idx_to_user[nb_idx]
        w = sims[nb_idx]
        if w <= 0:
            continue
        nb_ratings = user_ratings.get(nb_user, {})
        for b, r in nb_ratings.items():
            if b in rated_books:
                continue
            scores_sum[b] = scores_sum.get(b, 0.0) + w * r
            weights_sum[b] = weights_sum.get(b, 0.0) + w

    if not scores_sum:
        return []

    scores = {b: scores_sum[b] / (weights_sum[b] + 1e-9) for b in scores_sum}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [b for b, _ in ranked[:top_k]]


# ---------- SVD user-based scoring ----------

def recommend_svd_for_user(user_id, top_k, A):
    """
    SVD recommendation:
    - predict items closest to user in latent factor space (cosine).
    """
    user_to_idx = A["user_to_idx"]
    user_embed_np = A["user_embed_np"]
    item_embed_np = A["item_embed_np"]
    item_norms = A["item_norms"]
    user_ratings = A["user_ratings"]
    book_to_idx = A["book_to_idx"]

    if user_id not in user_to_idx:
        return []

    uid = user_to_idx[user_id]
    u_vec = user_embed_np[uid]
    u_norm = np.linalg.norm(u_vec) + 1e-9

    scores = (item_embed_np @ u_vec) / (item_norms * u_norm)

    rated = set(user_ratings.get(user_id, {}).keys())

    # map idx -> book_id
    idx_to_book = {v: k for k, v in book_to_idx.items()}

    candidates = []
    for idx, s in enumerate(scores):
        bid = idx_to_book[idx]
        if bid in rated:
            continue
        candidates.append((bid, s))

    ranked = sorted(candidates, key=lambda x: x[1], reverse=True)
    return [b for b, _ in ranked[:top_k]]


# ---------- CBF (TF-IDF) user-based scoring ----------

def recommend_cbf_for_user(user_id, top_k, A, rating_threshold=4.0):
    """
    Content-Based Filtering:
    - Build a user profile as the mean TF-IDF vector of liked books.
    - Recommend books with highest cosine similarity.
    """
    user_ratings = A["user_ratings"]
    if user_id not in user_ratings:
        return []

    cbf_content = A["cbf_content"]
    cbf_book_ids = A["cbf_book_ids"]
    cbf_book_to_row = A["cbf_book_to_row"]

    ratings = user_ratings[user_id]

    liked_books = [b for b, r in ratings.items() if r >= rating_threshold and b in cbf_book_to_row]
    if not liked_books:
        # Fallback: use all rated books
        liked_books = [b for b in ratings.keys() if b in cbf_book_to_row]

    if not liked_books:
        return []

    liked_rows = [cbf_book_to_row[b] for b in liked_books]
    user_vec = np.asarray(cbf_content[liked_rows].mean(axis=0)).reshape(1, -1)


    sims = cosine_similarity(user_vec, cbf_content).ravel()  # (num_books,)

    rated = set(ratings.keys())
    candidates = []
    for i, s in enumerate(sims):
        bid = cbf_book_ids[i]
        if bid in rated:
            continue
        candidates.append((bid, s))

    ranked = sorted(candidates, key=lambda x: x[1], reverse=True)
    return [b for b, _ in ranked[:top_k]]


# ---------- Pure RL, Item-CF, Hybrid wrappers ----------

def recommend_itemcf_for_user(user_id, top_k, A):
    """User-level Item-CF: score unseen books by similarity to user's history."""
    user_ratings = A["user_ratings"]
    user_hist = A["user_hist"]
    book_to_idx = A["book_to_idx"]

    if user_id not in user_ratings:
        return []

    rated = set(user_ratings[user_id].keys())
    all_books = set(book_to_idx.keys())
    candidates = list(all_books - rated)

    history = user_hist.get(user_id, [])
    cf_scores = score_candidates_with_itemcf(history, candidates, A)
    ranked = sorted(cf_scores.items(), key=lambda x: x[1], reverse=True)
    return [b for b, _ in ranked[:top_k]]


def recommend_rl_for_user(user_id, top_k, A):
    """Pure RL recommendations over unseen items."""
    user_ratings = A["user_ratings"]
    book_to_idx = A["book_to_idx"]

    if user_id not in user_ratings:
        return []

    rated = set(user_ratings[user_id].keys())
    all_books = set(book_to_idx.keys())
    candidates = list(all_books - rated)

    rl_scores = score_with_rl(user_id, candidates, A)
    ranked = sorted(zip(candidates, rl_scores), key=lambda x: x[1], reverse=True)
    return [b for b, _ in ranked[:top_k]]


def recommend_hybrid_for_user(user_id, top_k, A):
    """
    Hybrid (your best model):
    Z-score(CF) and RL, then blend:
        score = CF_WEIGHT * CF_z + (1 - CF_WEIGHT) * RL_z
    """
    user_ratings = A["user_ratings"]
    user_hist = A["user_hist"]
    book_to_idx = A["book_to_idx"]
    CF_WEIGHT = A["CF_WEIGHT"]

    if user_id not in user_ratings:
        return []

    rated = set(user_ratings[user_id].keys())
    all_books = set(book_to_idx.keys())
    candidates = list(all_books - rated)

    # CF part
    history = user_hist.get(user_id, [])
    cf_scores = score_candidates_with_itemcf(history, candidates, A)
    cf_z = zscore_dict(cf_scores)

    # RL part
    rl_scores = score_with_rl(user_id, candidates, A)
    rl_z = zscore_array(rl_scores)

    blended = {
        b: CF_WEIGHT * cf_z.get(b, 0.0) + (1 - CF_WEIGHT) * rl_z[i]
        for i, b in enumerate(candidates)
    }
    ranked = sorted(blended.items(), key=lambda x: x[1], reverse=True)
    return [b for b, _ in ranked[:top_k]]


# ============================================================
# ITEM-BASED RECS FOR A GIVEN BOOK
# ============================================================

def similar_itemcf_books(book_id, top_k, A):
    """Item-CF similar books for a given book_id."""
    book_to_idx = A["book_to_idx"]
    idx_to_book = A["idx_to_book"]
    item_sim = A["item_sim"]

    bi = book_to_idx.get(book_id)
    if bi is None:
        return []

    sims = item_sim[bi]
    order = np.argsort(sims)[::-1]

    res = []
    for idx in order:
        candidate = idx_to_book[idx]
        if candidate == book_id:
            continue
        res.append(candidate)
        if len(res) >= top_k:
            break
    return res


def similar_svd_books(book_id, top_k, A):
    """SVD-embedding similarity for items."""
    book_to_idx = A["book_to_idx"]
    idx_to_book = A["idx_to_book"]
    item_embed_np = A["item_embed_np"]
    item_norms = A["item_norms"]

    bi = book_to_idx.get(book_id)
    if bi is None:
        return []

    v = item_embed_np[bi]
    v_norm = np.linalg.norm(v) + 1e-9

    scores = (item_embed_np @ v) / (item_norms * v_norm)
    order = np.argsort(scores)[::-1]

    res = []
    for idx in order:
        candidate = idx_to_book[idx]
        if candidate == book_id:
            continue
        res.append(candidate)
        if len(res) >= top_k:
            break
    return res


def similar_cbf_books(book_id, top_k, A):
    """
    Content-Based (TF-IDF) similarity:
    - Cosine similarity over TF-IDF(title + authors).
    """
    cbf_book_to_row = A["cbf_book_to_row"]
    cbf_book_ids = A["cbf_book_ids"]
    cbf_content = A["cbf_content"]

    if book_id not in cbf_book_to_row:
        return []

    row_idx = cbf_book_to_row[book_id]
    v = cbf_content[row_idx]

    sims = cosine_similarity(v, cbf_content).ravel()
    order = np.argsort(sims)[::-1]

    res = []
    for idx in order:
        candidate = cbf_book_ids[idx]
        if candidate == book_id:
            continue
        res.append(candidate)
        if len(res) >= top_k:
            break
    return res


def similar_rl_books(book_id, top_k, A):
    """
    RL-based similarity for a book:
    - Uses average user embedding as a generic RL state.
    - Scores all items by Q(avg_user, item).
    - Returns top-K books (excluding the query book).
    """
    device = A["device"]
    qnet = A["qnet"]

    book_to_idx = A["book_to_idx"]
    idx_to_book = A["idx_to_book"]

    item_embed = A["item_embed_t"]
    cf_vecs = A["cf_vecs"]
    user_embed = A["user_embed_t"]

    if book_id not in book_to_idx:
        return []

    # Generic user state: average user embedding
    avg_user = user_embed.mean(dim=0)
    state_vec = torch.cat([avg_user, torch.zeros(50, device=device)])  # (150,)

    item_vecs = []
    for bi in range(len(book_to_idx)):
        item_vec = item_embed[bi]
        cf_vec = cf_vecs[bi]
        full_vec = torch.cat([item_vec, cf_vec])
        item_vecs.append(full_vec)

    V = torch.stack(item_vecs)                            # (N, 150)
    S = state_vec.unsqueeze(0).expand(len(V), -1)         # (N, 150)
    SA = S * V                                            # (N, 150)

    with torch.no_grad():
        q_vals = qnet(SA).squeeze(1).cpu().numpy()        # (N,)

    order = np.argsort(q_vals)[::-1]

    res = []
    for idx in order:
        candidate = idx_to_book[idx]
        if candidate == book_id:
            continue
        res.append(candidate)
        if len(res) >= top_k:
            break
    return res


def similar_hybrid_books(book_id, top_k, A, pool_size=200):
    """
    Hybrid item-level similarity:
    - Combines Item-CF rank and RL-based rank via Z-score blending.
    - Uses the same CF_WEIGHT as the user-level hybrid model.
    """
    CF_WEIGHT = A["CF_WEIGHT"]

    icf_list = similar_itemcf_books(book_id, top_k=pool_size, A=A)
    rl_list = similar_rl_books(book_id, top_k=pool_size, A=A)

    icf_score = {b: (len(icf_list) - i) for i, b in enumerate(icf_list)}
    rl_score = {b: (len(rl_list) - i) for i, b in enumerate(rl_list)}

    icf_z = zscore_dict(icf_score)
    rl_z = zscore_dict(rl_score)

    all_books = set(icf_score.keys()) | set(rl_score.keys())

    blended = {
        b: CF_WEIGHT * icf_z.get(b, 0.0) + (1 - CF_WEIGHT) * rl_z.get(b, 0.0)
        for b in all_books
    }
    ranked = sorted(blended.items(), key=lambda x: x[1], reverse=True)
    return [b for b, _ in ranked[:top_k]]


# ============================================================
# PRINT HELPERS
# ============================================================

def print_book(bid, meta):
    m = meta.get(bid, {})
    print(f"  [{bid}] {m.get('title', 'Unknown')} — {m.get('authors', 'Unknown')}")


def print_book_list(title, books, meta):
    print("\n" + title)
    print("-" * len(title))
    if not books:
        print("  (no results)")
        return
    for b in books:
        print_book(b, meta)
    print()


# ============================================================
# COMPARISON VIEW FOR A USER
# ============================================================

def show_all_models_for_user(user_id, A, top_k=10):
    meta = A["book_meta"]

    print(f"\n🔎 Recommendations for user_id = {user_id}")

    # User-CF
    ucf = recommend_usercf_for_user(user_id, top_k, A)
    print_book_list("User-CF Recommendations:", ucf, meta)

    # Item-CF
    icf = recommend_itemcf_for_user(user_id, top_k, A)
    print_book_list("Item-CF Recommendations:", icf, meta)

    # SVD
    svd_recs = recommend_svd_for_user(user_id, top_k, A)
    print_book_list("SVD Recommendations:", svd_recs, meta)

    # CBF
    cbf_recs = recommend_cbf_for_user(user_id, top_k, A)
    print_book_list("Content-Based (TF-IDF) Recommendations:", cbf_recs, meta)

    # RL
    rl_recs = recommend_rl_for_user(user_id, top_k, A)
    print_book_list("RL-Only Recommendations:", rl_recs, meta)

    # Hybrid
    hybrid = recommend_hybrid_for_user(user_id, top_k, A)
    print_book_list("Hybrid (CF + RL, Z-score) Recommendations:", hybrid, meta)


# ============================================================
# MENU ACTIONS
# ============================================================

def action_random_user(A):
    """Pick random user and show recs from all models."""
    user_ratings = A["user_ratings"]
    meta = A["book_meta"]

    u = random.choice(list(user_ratings.keys()))
    print(f"\n🎯 Random user selected: {u}")

    # show user's top-rated
    ratings = user_ratings[u]
    top_r = sorted(ratings.items(), key=lambda x: x[1], reverse=True)[:5]
    top_ids = [b for b, _ in top_r]
    print_book_list("User's Top Rated Books:", top_ids, meta)

    show_all_models_for_user(u, A)


def action_book_search(A):
    """
    Fuzzy book title search, then show Item-CF / SVD / RL / Hybrid / CBF similar books.
    """
    meta = A["book_meta"]

    query = input("\nEnter a book title: ").strip()
    if not query:
        print("❌ Empty search.")
        return

    book_ids = list(meta.keys())
    titles = [meta[b]["title"] for b in book_ids]

    matches = process.extract(query, titles, scorer=fuzz.WRatio, limit=3)
    if not matches:
        print("No close title matches found.")
        return

    print("\nClosest matches:")
    print("----------------")
    for i, (title, score, idx) in enumerate(matches, start=1):
        bid = book_ids[idx]
        print(f"{i}. {title} (book_id={bid}, score={score:.1f}%)")

    choice = input("\nSelect 1/2/3 (Enter = 1): ").strip()
    if choice == "":
        sel = 0
    elif choice in {"1", "2", "3"}:
        sel = int(choice) - 1
    else:
        print("Invalid selection.")
        return

    _, _, pos = matches[sel]
    selected_book = book_ids[pos]

    print("\nYou selected:")
    print_book(selected_book, meta)

    # Item-CF
    icf = similar_itemcf_books(selected_book, top_k=10, A=A)
    print_book_list("Item-CF Similar Books:", icf, meta)

    # SVD
    svd_sim = similar_svd_books(selected_book, top_k=10, A=A)
    print_book_list("SVD-Embedding Similar Books:", svd_sim, meta)

    # RL
    rl_sim = similar_rl_books(selected_book, top_k=10, A=A)
    print_book_list("RL-Based Similar Books (avg user state):", rl_sim, meta)

    # Hybrid
    hybrid_sim = similar_hybrid_books(selected_book, top_k=10, A=A)
    print_book_list("Hybrid (Item-CF + RL) Similar Books:", hybrid_sim, meta)

    # Content-Based
    cbf_sim = similar_cbf_books(selected_book, top_k=10, A=A)
    print_book_list("Content-Based (TF-IDF) Similar Books:", cbf_sim, meta)


def action_user_input(A):
    """User enters user_id, then show all model outputs."""
    try:
        u = int(input("\nEnter user_id: ").strip())
    except ValueError:
        print("Invalid user_id (must be int).")
        return

    show_all_models_for_user(u, A)


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    A = load_artifacts()

    while True:
        print("\n==============================")
        print("📚 BOOK RECOMMENDER SYSTEM")
        print("==============================")
        print("1. Random user → compare all models")
        print("2. Enter a book title → compare Item-CF / SVD / RL / Hybrid / CBF")
        print("3. Enter a user ID → compare all models")
        print("4. Exit")

        choice = input("\nChoose an option: ").strip()

        if choice == "1":
            action_random_user(A)
        elif choice == "2":
            action_book_search(A)
        elif choice == "3":
            action_user_input(A)
        elif choice == "4":
            print("\nGoodbye!\n")
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()

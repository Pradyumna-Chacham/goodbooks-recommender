# src/goodbooks_rec/multimodel.py
"""
Deployment-friendly multi-model recommenders (Top-K CF)

- Uses Top-K neighbor artifacts instead of dense NxN similarity matrices
- Loads everything once (wrap load_mm_artifacts with @st.cache_resource in Streamlit)
- Unified interface: recommend_for_user(model_name, user_id, k, A)

Expected artifacts:
  models/svd_hybrid.pkl
  models/rl_cfhard_fast.pth
  models/hybrid_reranker.pkl
  models/item_topk_k100.npz   (idx: [n_items,K], sim: [n_items,K])
  models/user_topk_k100.npz   (idx: [n_users,K], sim: [n_users,K])
  new_data/train_df.pkl
  data/book_metadata.csv
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Paths
# -----------------------------
REPO_ROOT = Path(".")
MODELS_DIR = REPO_ROOT / "models"
DATA_DIR = REPO_ROOT / "new_data"
RAW_DIR = REPO_ROOT / "data"

SVD_HYBRID_PATH = MODELS_DIR / "svd_hybrid.pkl"
RL_WEIGHTS_PATH = MODELS_DIR / "rl_cfhard_fast.pth"
HYBRID_RERANKER_PATH = MODELS_DIR / "hybrid_reranker.pkl"

ITEM_TOPK_PATH = MODELS_DIR / "item_topk_k100.npz"
USER_TOPK_PATH = MODELS_DIR / "user_topk_k100.npz"

TRAIN_DF_PATH = DATA_DIR / "train_df.pkl"
BOOK_META_CSV = RAW_DIR / "book_metadata.csv"


MODEL_NAMES = [
    "Popular",
    "User-CF",
    "Item-CF",
    "SVD",
    "RL",
    "Hybrid",
    "Content-Based",
]


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _zscore_dict(d: Dict[int, float]) -> Dict[int, float]:
    if not d:
        return {}
    vals = np.array(list(d.values()), dtype=np.float32)
    mu = float(vals.mean())
    sd = float(vals.std() + 1e-9)
    return {k: (float(v) - mu) / sd for k, v in d.items()}


def _load_torch_and_qnet():
    import torch
    import torch.nn as nn

    class QNet(nn.Module):
        def __init__(self, dim: int = 150):
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

    return torch, QNet


def load_mm_artifacts(
    item_topk_path: Path = ITEM_TOPK_PATH,
    user_topk_path: Path = USER_TOPK_PATH,
    topk_for_rl: int = 50,
    svd_path: Path = SVD_HYBRID_PATH,
    train_df_path: Path = TRAIN_DF_PATH,
    book_meta_csv: Path = BOOK_META_CSV,
    hybrid_path: Path = HYBRID_RERANKER_PATH,
    rl_weights_path: Path = RL_WEIGHTS_PATH,
    load_rl: bool = True,
) -> Dict:
    """
    Loads all deployable artifacts for all models.
    Wrap this in @st.cache_resource in Streamlit.
    """
    # ---- SVD embeddings + mappings
    svd = _load_pickle(svd_path)
    user_embed_np = np.asarray(svd["user_embed"], dtype=np.float32)  # [n_users, d]
    item_embed_np = np.asarray(svd["item_embed"], dtype=np.float32)  # [n_items, d]
    user_to_idx: Dict[int, int] = svd["user_to_idx"]
    book_to_idx: Dict[int, int] = svd["book_to_idx"]
    idx_to_book = {v: k for k, v in book_to_idx.items()}
    idx_to_user = {v: k for k, v in user_to_idx.items()}
    item_norms = np.linalg.norm(item_embed_np, axis=1).astype(np.float32) + 1e-9

    # ---- Top-K neighbors
    item_npz = np.load(item_topk_path)
    item_topk_idx = item_npz["idx"].astype(np.int32, copy=False)
    item_topk_sim = item_npz["sim"].astype(np.float32, copy=False)

    user_npz = np.load(user_topk_path)
    user_topk_idx = user_npz["idx"].astype(np.int32, copy=False)
    user_topk_sim = user_npz["sim"].astype(np.float32, copy=False)

    # ---- interactions: seen + popularity + ratings
    train_df = pd.read_pickle(train_df_path)
    train_df["user_id"] = train_df["user_id"].astype(int)
    train_df["book_id"] = train_df["book_id"].astype(int)

    user_hist = train_df.groupby("user_id")["book_id"].apply(list).to_dict()
    user_ratings = (
        train_df.groupby("user_id")[["book_id", "rating"]]
        .apply(lambda df: dict(zip(df["book_id"], df["rating"])))
        .to_dict()
    )

    pop = train_df.groupby("book_id").agg(
        n_ratings=("rating", "size"),
        mean_rating=("rating", "mean"),
    )
    pop["score"] = pop["n_ratings"] * pop["mean_rating"]
    pop_sorted = pop.sort_values("score", ascending=False).index.astype(int).tolist()

    # ---- content-based TF-IDF (title + authors)
    meta_df = pd.read_csv(book_meta_csv)
    meta_df["book_id"] = meta_df["book_id"].astype(int)

    book_meta = {
        int(row.book_id): {"title": str(row.title), "authors": str(row.authors)}
        for _, row in meta_df.iterrows()
    }

    cbf_book_ids = meta_df["book_id"].astype(int).tolist()
    cbf_texts = [f'{book_meta[b]["title"]} {book_meta[b]["authors"]}' for b in cbf_book_ids]
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    cbf_matrix = tfidf.fit_transform(cbf_texts)
    cbf_book_to_row = {bid: i for i, bid in enumerate(cbf_book_ids)}

    # ---- hybrid config
    hybrid_cfg = _load_pickle(hybrid_path)
    CF_WEIGHT = float(hybrid_cfg.get("CF_WEIGHT", 0.5))

    # ---- RL load (optional)
    if load_rl:
        torch, QNet = _load_torch_and_qnet()
        device = torch.device("cpu")

        user_embed_t = torch.tensor(user_embed_np, dtype=torch.float32).to(device)
        item_embed_t = torch.tensor(item_embed_np, dtype=torch.float32).to(device)

        n_items = item_embed_np.shape[0]
        cf_vecs_np = np.zeros((n_items, 50), dtype=np.float32)
        use_k = min(topk_for_rl, item_topk_idx.shape[1])

        for i in range(n_items):
            nbrs = item_topk_idx[i, :use_k]
            cf_vecs_np[i] = item_embed_np[nbrs].mean(axis=0)[:50]

        cf_vecs_t = torch.tensor(cf_vecs_np, dtype=torch.float32).to(device)

        qnet = QNet(dim=150).to(device)
        state = torch.load(rl_weights_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            qnet.load_state_dict(state["state_dict"])
        else:
            qnet.load_state_dict(state)
        qnet.eval()
    else:
        torch = None
        device = None
        qnet = None
        user_embed_t = None
        item_embed_t = None
        cf_vecs_t = None

    return {
        "user_to_idx": user_to_idx,
        "book_to_idx": book_to_idx,
        "idx_to_book": idx_to_book,
        "idx_to_user": idx_to_user,
        "user_embed_np": user_embed_np,
        "item_embed_np": item_embed_np,
        "item_norms": item_norms,
        "item_topk_idx": item_topk_idx,
        "item_topk_sim": item_topk_sim,
        "user_topk_idx": user_topk_idx,
        "user_topk_sim": user_topk_sim,
        "train_df": train_df,
        "user_hist": user_hist,
        "user_ratings": user_ratings,
        "pop_sorted": pop_sorted,
        "book_meta": book_meta,
        "cbf_book_ids": cbf_book_ids,
        "cbf_book_to_row": cbf_book_to_row,
        "cbf_matrix": cbf_matrix,
        "tfidf": tfidf,
        "device": device,
        "torch": torch,
        "qnet": qnet,
        "user_embed_t": user_embed_t,
        "item_embed_t": item_embed_t,
        "cf_vecs_t": cf_vecs_t,
        "CF_WEIGHT": CF_WEIGHT,
    }


def recommend_popular_user(user_id: int, k: int, A: Dict) -> List[int]:
    seen = set(A["user_hist"].get(user_id, []))
    out: List[int] = []
    for b in A["pop_sorted"]:
        if b not in seen:
            out.append(int(b))
        if len(out) >= k:
            break

    return out


def recommend_user_cf(user_id: int, k: int, A: Dict) -> List[int]:
    user_to_idx = A["user_to_idx"]
    if user_id not in user_to_idx:
        return recommend_popular_user(user_id, k, A)

    uidx = user_to_idx[user_id]
    nbr_idxs = A["user_topk_idx"][uidx]
    nbr_sims = A["user_topk_sim"][uidx]

    seen = set(A["user_hist"].get(user_id, []))
    scores: Dict[int, float] = {}

    idx_to_user = A["idx_to_user"]
    user_ratings = A["user_ratings"]

    for nidx, w in zip(nbr_idxs, nbr_sims):
        nu = idx_to_user.get(int(nidx))
        if nu is None:
            continue
        wr = float(w)
        for b, r in user_ratings.get(nu, {}).items():
            bi = int(b)
            if bi in seen:
                continue
            scores[bi] = scores.get(bi, 0.0) + wr * float(r)

    if not scores:
        return recommend_popular_user(user_id, k, A)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [b for b, _ in ranked[:k]]


def recommend_item_cf(user_id: int, k: int, A: Dict) -> List[int]:
    hist = A["user_hist"].get(user_id)
    if not hist:
        return recommend_popular_user(user_id, k, A)

    book_to_idx = A["book_to_idx"]
    idx_to_book = A["idx_to_book"]
    item_topk_idx = A["item_topk_idx"]
    item_topk_sim = A["item_topk_sim"]

    seen = set(hist)
    ratings = A["user_ratings"].get(user_id, {})
    scores: Dict[int, float] = {}

    for b in hist[-50:]:
        bi = book_to_idx.get(int(b))
        if bi is None:
            continue

        r = float(ratings.get(int(b), 3.0))
        w_r = max(-1.0, min(1.0, (r - 2.5) / 2.5))

        nbrs = item_topk_idx[bi]
        sims = item_topk_sim[bi]

        for nidx, sim in zip(nbrs, sims):
            cand_book = idx_to_book.get(int(nidx))
            if cand_book is None:
                continue
            cand_book = int(cand_book)
            if cand_book in seen:
                continue
            scores[cand_book] = scores.get(cand_book, 0.0) + float(sim) * w_r

    if not scores:
        return recommend_popular_user(user_id, k, A)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [b for b, _ in ranked[:k]]


def recommend_svd(user_id: int, k: int, A: Dict) -> List[int]:
    user_to_idx = A["user_to_idx"]
    if user_id not in user_to_idx:
        return recommend_popular_user(user_id, k, A)

    uidx = user_to_idx[user_id]
    u = A["user_embed_np"][uidx]
    V = A["item_embed_np"]
    norms = A["item_norms"]
    u_norm = float(np.linalg.norm(u) + 1e-9)

    scores = (V @ u) / (norms * u_norm)

    seen = set(A["user_hist"].get(user_id, []))
    order = np.argsort(scores)[::-1]

    out: List[int] = []
    for j in order:
        b = A["idx_to_book"].get(int(j))
        if b is None:
            continue
        b = int(b)
        if b in seen:
            continue
        out.append(b)
        if len(out) >= k:
            break
    return out


def _score_with_rl(user_id: int, candidates: List[int], A: Dict) -> np.ndarray:
    user_to_idx = A["user_to_idx"]
    book_to_idx = A["book_to_idx"]
    if user_id not in user_to_idx:
        return np.zeros(len(candidates), dtype=np.float32)

    torch = A["torch"]
    device = A["device"]
    qnet = A["qnet"]
    user_embed_t = A["user_embed_t"]
    item_embed_t = A["item_embed_t"]
    cf_vecs_t = A["cf_vecs_t"]

    uid = user_to_idx[user_id]
    state = torch.cat([user_embed_t[uid], torch.zeros(50, device=device)])  # (150,)

    item_vecs = []
    valid_pos = []

    for i, b in enumerate(candidates):
        bi = book_to_idx.get(int(b))
        if bi is None:
            continue
        it = torch.cat([item_embed_t[bi], cf_vecs_t[bi]])  # (150,)
        item_vecs.append(it)
        valid_pos.append(i)

    out = np.zeros(len(candidates), dtype=np.float32)
    if not item_vecs:
        return out

    X = torch.stack(item_vecs, dim=0)
    state_rep = state.unsqueeze(0).expand(X.shape[0], -1)
    inp = state_rep * X

    with torch.no_grad():
        q = qnet(inp).squeeze(-1).cpu().numpy().astype(np.float32)

    for p, score in zip(valid_pos, q):
        out[p] = float(score)
    return out


def recommend_rl(user_id: int, k: int, A: Dict) -> List[int]:
    shortlist = recommend_svd(user_id, k=1000, A=A)
    if not shortlist:
        return recommend_popular_user(user_id, k, A)

    scores = _score_with_rl(user_id, shortlist, A)
    order = np.argsort(scores)[::-1]
    return [shortlist[i] for i in order[:k]]


def recommend_hybrid(user_id: int, k: int, A: Dict) -> List[int]:
    hist = A["user_hist"].get(user_id)
    if not hist:
        return recommend_popular_user(user_id, k, A)

    candidates = recommend_item_cf(user_id, k=2000, A=A)
    if not candidates:
        return recommend_popular_user(user_id, k, A)

    book_to_idx = A["book_to_idx"]
    idx_to_book = A["idx_to_book"]
    item_topk_idx = A["item_topk_idx"]
    item_topk_sim = A["item_topk_sim"]

    cand_set = set(candidates)
    seen = set(hist)
    ratings = A["user_ratings"].get(user_id, {})

    cf_scores: Dict[int, float] = {c: 0.0 for c in candidates}

    for b in hist[-50:]:
        bi = book_to_idx.get(int(b))
        if bi is None:
            continue
        r = float(ratings.get(int(b), 3.0))
        w_r = max(-1.0, min(1.0, (r - 2.5) / 2.5))

        nbrs = item_topk_idx[bi]
        sims = item_topk_sim[bi]

        for nidx, sim in zip(nbrs, sims):
            cand_book = idx_to_book.get(int(nidx))
            if cand_book is None:
                continue
            cand_book = int(cand_book)
            if cand_book in seen or cand_book not in cand_set:
                continue
            cf_scores[cand_book] += float(sim) * w_r

    rl_arr = _score_with_rl(user_id, candidates, A)
    rl_scores = {candidates[i]: float(rl_arr[i]) for i in range(len(candidates))}

    cf_z = _zscore_dict(cf_scores)
    rl_z = _zscore_dict(rl_scores)

    w = float(A["CF_WEIGHT"])
    blend = {b: w * cf_z.get(b, 0.0) + (1.0 - w) * rl_z.get(b, 0.0) for b in candidates}

    ranked = sorted(blend.items(), key=lambda x: x[1], reverse=True)
    return [b for b, _ in ranked[:k]]


def recommend_cbf(user_id: int, k: int, A: Dict) -> List[int]:
    seen = set(A["user_hist"].get(user_id, []))
    ratings = A["user_ratings"].get(user_id, {})

    seeds = [int(b) for b, r in ratings.items() if float(r) >= 4.0]
    if not seeds:
        seeds = [b for b in A["pop_sorted"] if b in A["cbf_book_to_row"]][:5]

    rows = [A["cbf_book_to_row"].get(b) for b in seeds]
    rows = [r for r in rows if r is not None]
    if not rows:
        return recommend_popular_user(user_id, k, A)

    user_vec = A["cbf_matrix"][rows].mean(axis=0)
    user_vec = np.asarray(user_vec)  # converts np.matrix → ndarray
    # ensure 2D shape (1, n_features)
    if user_vec.ndim == 1:
        user_vec = user_vec.reshape(1, -1)
    sims = cosine_similarity(user_vec, A["cbf_matrix"]).ravel()

    order = np.argsort(sims)[::-1]
    out: List[int] = []
    for j in order:
        b = int(A["cbf_book_ids"][int(j)])
        if b in seen:
            continue
        out.append(b)
        if len(out) >= k:
            break
    return out


def recommend_for_user(model_name: str, user_id: int, k: int, A: Dict) -> List[int]:
    model_name = str(model_name)

    if model_name == "Popular":
        return recommend_popular_user(user_id, k, A)
    if model_name == "User-CF":
        return recommend_user_cf(user_id, k, A)
    if model_name == "Item-CF":
        return recommend_item_cf(user_id, k, A)
    if model_name == "SVD":
        return recommend_svd(user_id, k, A)
    if model_name == "RL":
        return recommend_rl(user_id, k, A)
    if model_name == "Hybrid":
        return recommend_hybrid(user_id, k, A)
    if model_name == "Content-Based":
        return recommend_cbf(user_id, k, A)

    return recommend_popular_user(user_id, k, A)

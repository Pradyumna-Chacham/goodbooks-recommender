from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

FIXTURES = Path(__file__).parent.parent / "fixtures" / "mini"


def build_mini_A_for_multimodel(k: int = 10) -> dict:
    # --- load mini svd bundle
    with open(FIXTURES / "svd_hybrid.pkl", "rb") as f:
        svd = pickle.load(f)

    user_embed_np = np.asarray(svd["user_embed"], dtype=np.float32)
    item_embed_np = np.asarray(svd["item_embed"], dtype=np.float32)
    user_to_idx = svd["user_to_idx"]
    book_to_idx = svd["book_to_idx"]

    idx_to_user = {v: k for k, v in user_to_idx.items()}
    idx_to_book = {v: k for k, v in book_to_idx.items()}
    item_norms = np.linalg.norm(item_embed_np, axis=1).astype(np.float32) + 1e-9

    # --- load topk
    item_npz = np.load(FIXTURES / f"item_topk_k{k}.npz")
    user_npz = np.load(FIXTURES / f"user_topk_k{k}.npz")

    item_topk_idx = item_npz["idx"].astype(np.int32, copy=False)
    item_topk_sim = item_npz["sim"].astype(np.float32, copy=False)
    user_topk_idx = user_npz["idx"].astype(np.int32, copy=False)
    user_topk_sim = user_npz["sim"].astype(np.float32, copy=False)

    # --- load interactions
    train_df = pd.read_pickle(FIXTURES / "train_df.pkl")
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

    # --- content metadata (mini)
    meta_df = pd.read_csv(FIXTURES / "book_metadata.csv")
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

    # --- minimal RL placeholders (tests will monkeypatch _score_with_rl)
    # these keys must exist for hybrid/RL paths, but we won't actually run torch.
    A = {
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
        "CF_WEIGHT": 0.5,
        # RL keys (not used if we monkeypatch _score_with_rl)
        "torch": None,
        "device": None,
        "qnet": None,
        "user_embed_t": None,
        "item_embed_t": None,
        "cf_vecs_t": None,
    }
    return A

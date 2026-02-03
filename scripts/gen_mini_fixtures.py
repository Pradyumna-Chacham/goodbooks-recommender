#!/usr/bin/env python3
"""
Generate mini test fixtures for goodbooks-rec.

Outputs (default):
  tests/fixtures/mini/
    svd_hybrid.pkl
    item_topk_k10.npz
    user_topk_k10.npz
    book_metadata.csv
    train_df.pkl

This uses your *real* production artifacts as the source:
  models/svd_hybrid.pkl
  data/book_metadata.csv
  new_data/train_df.pkl (if present) else data/ratings.csv

Run:
  python scripts/make_mini_fixtures.py
  python scripts/make_mini_fixtures.py --n_users 40 --n_items 200 --k 10 --seed 42
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _save_pickle(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _cosine_topk(X: np.ndarray, k: int, block: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cosine Top-K neighbors for rows of X (within X), excluding self.

    Returns:
      idx: int32 [n, k]
      sim: float32 [n, k]
    """
    X = np.asarray(X, dtype=np.float32)
    n, d = X.shape
    if k >= n:
        raise ValueError(f"k={k} must be < n={n} for self-excluding topk.")

    # Normalize for cosine
    norms = np.linalg.norm(X, axis=1, keepdims=True).astype(np.float32) + 1e-9
    Xn = X / norms

    idx_out = np.empty((n, k), dtype=np.int32)
    sim_out = np.empty((n, k), dtype=np.float32)

    # Compute in blocks to avoid large n*n allocations when n grows.
    # For typical mini sizes (<= 500), this is still very fast.
    for start in range(0, n, block):
        end = min(n, start + block)
        S = Xn[start:end] @ Xn.T  # [b, n]
        # Exclude self similarities
        for i in range(start, end):
            S[i - start, i] = -np.inf

        # Partial topk (unordered), then sort within topk by sim desc
        part = np.argpartition(-S, kth=np.arange(k), axis=1)[:, :k]  # [b, k]
        sims = np.take_along_axis(S, part, axis=1)  # [b, k]
        order = np.argsort(-sims, axis=1)
        top_idx = np.take_along_axis(part, order, axis=1)
        top_sim = np.take_along_axis(sims, order, axis=1)

        idx_out[start:end] = top_idx.astype(np.int32, copy=False)
        sim_out[start:end] = top_sim.astype(np.float32, copy=False)

    return idx_out, sim_out


def _sample_keys(keys: List[int], n: int, rng: np.random.Generator) -> List[int]:
    if n > len(keys):
        raise ValueError(f"Requested n={n} but only {len(keys)} available.")
    # deterministic random sample
    chosen = rng.choice(np.array(keys, dtype=np.int64), size=n, replace=False)
    return chosen.astype(int).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_users", type=int, default=50)
    ap.add_argument("--n_items", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out_dir", type=str, default="tests/fixtures/mini")

    ap.add_argument("--svd_path", type=str, default="models/svd_hybrid.pkl")
    ap.add_argument("--meta_csv", type=str, default="data/book_metadata.csv")
    ap.add_argument("--train_df_pkl", type=str, default="new_data/train_df.pkl")
    ap.add_argument("--ratings_csv", type=str, default="data/ratings.csv")

    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    svd_path = Path(args.svd_path)
    meta_csv = Path(args.meta_csv)
    train_df_pkl = Path(args.train_df_pkl)
    ratings_csv = Path(args.ratings_csv)

    print(f"[mini-fixtures] Loading SVD artifact: {svd_path}")
    svd = _load_pickle(svd_path)

    user_embed = np.asarray(svd["user_embed"], dtype=np.float32)
    item_embed = np.asarray(svd["item_embed"], dtype=np.float32)
    user_to_idx: Dict[int, int] = svd["user_to_idx"]
    book_to_idx: Dict[int, int] = svd["book_to_idx"]

    # Sample users and items by *IDs*
    all_user_ids = list(user_to_idx.keys())
    all_book_ids = list(book_to_idx.keys())

    mini_user_ids = _sample_keys(all_user_ids, args.n_users, rng)
    mini_book_ids = _sample_keys(all_book_ids, args.n_items, rng)

    # Build new compact mappings
    mini_user_to_idx = {uid: i for i, uid in enumerate(mini_user_ids)}
    mini_book_to_idx = {bid: i for i, bid in enumerate(mini_book_ids)}

    # Subset embeddings in the same order as mini_*_ids
    mini_user_idx_old = np.array([user_to_idx[uid] for uid in mini_user_ids], dtype=np.int64)
    mini_book_idx_old = np.array([book_to_idx[bid] for bid in mini_book_ids], dtype=np.int64)

    mini_user_embed = user_embed[mini_user_idx_old]
    mini_item_embed = item_embed[mini_book_idx_old]

    # Save mini SVD pickle
    mini_svd = {
        "user_embed": mini_user_embed,
        "item_embed": mini_item_embed,
        "user_to_idx": mini_user_to_idx,
        "book_to_idx": mini_book_to_idx,
    }
    _save_pickle(mini_svd, out_dir / "svd_hybrid.pkl")
    print(f"[mini-fixtures] Wrote: {out_dir / 'svd_hybrid.pkl'}")

    # Compute Top-K neighbors within mini subset
    k = args.k
    if k >= args.n_items or k >= args.n_users:
        raise ValueError("k must be smaller than BOTH n_items and n_users for self-excluding TopK.")

    print(f"[mini-fixtures] Computing item TopK (n={args.n_items}, k={k}) ...")
    item_topk_idx, item_topk_sim = _cosine_topk(mini_item_embed, k=k)
    np.savez_compressed(out_dir / f"item_topk_k{k}.npz", idx=item_topk_idx, sim=item_topk_sim)
    print(f"[mini-fixtures] Wrote: {out_dir / f'item_topk_k{k}.npz'}")

    print(f"[mini-fixtures] Computing user TopK (n={args.n_users}, k={k}) ...")
    user_topk_idx, user_topk_sim = _cosine_topk(mini_user_embed, k=k)
    np.savez_compressed(out_dir / f"user_topk_k{k}.npz", idx=user_topk_idx, sim=user_topk_sim)
    print(f"[mini-fixtures] Wrote: {out_dir / f'user_topk_k{k}.npz'}")

    # Write metadata subset
    print(f"[mini-fixtures] Loading metadata: {meta_csv}")
    meta_df = pd.read_csv(meta_csv)
    meta_df["book_id"] = meta_df["book_id"].astype(int)

    keep_cols = [c for c in ["book_id", "title", "authors"] if c in meta_df.columns]
    mini_meta = meta_df.loc[meta_df["book_id"].isin(mini_book_ids), keep_cols].copy()

    # ensure all mini_book_ids appear (fill missing with placeholders)
    present = set(mini_meta["book_id"].astype(int).tolist())
    missing = [bid for bid in mini_book_ids if bid not in present]
    if missing:
        filler = pd.DataFrame(
            {
                "book_id": missing,
                "title": [f"Unknown title ({bid})" for bid in missing],
                "authors": ["Unknown" for _ in missing],
            }
        )
        mini_meta = pd.concat([mini_meta, filler], ignore_index=True)

    mini_meta.to_csv(out_dir / "book_metadata.csv", index=False)
    print(f"[mini-fixtures] Wrote: {out_dir / 'book_metadata.csv'}")

    # Optional: write a tiny train_df for "seen" / popularity tests
    if train_df_pkl.exists():
        print(f"[mini-fixtures] Loading train_df: {train_df_pkl}")
        train_df = pd.read_pickle(train_df_pkl)
        # normalize columns
        if "user_id" not in train_df.columns or "book_id" not in train_df.columns:
            raise ValueError("train_df.pkl must have columns user_id and book_id (and rating).")
        train_df["user_id"] = train_df["user_id"].astype(int)
        train_df["book_id"] = train_df["book_id"].astype(int)
    else:
        print(f"[mini-fixtures] train_df.pkl not found, falling back to ratings csv: {ratings_csv}")
        train_df = pd.read_csv(ratings_csv)
        # expected Goodbooks schema: user_id, book_id, rating
        train_df["user_id"] = train_df["user_id"].astype(int)
        train_df["book_id"] = train_df["book_id"].astype(int)

    mini_train = train_df[
        train_df["user_id"].isin(mini_user_ids) & train_df["book_id"].isin(mini_book_ids)
    ].copy()

    # Keep it small: cap rows (helps repo + CI)
    mini_train = mini_train.head(5000)

    # Ensure rating exists
    if "rating" not in mini_train.columns:
        mini_train["rating"] = 0.0

    mini_train.to_pickle(out_dir / "train_df.pkl")
    print(f"[mini-fixtures] Wrote: {out_dir / 'train_df.pkl'} (rows={len(mini_train)})")

    print("[mini-fixtures] Done.")


if __name__ == "__main__":
    main()

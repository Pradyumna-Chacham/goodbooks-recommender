"""
build_dataset_20k.py
---------------------

Builds a 20,000-user, genre-aware, dense recommendation dataset.

Requirements:
    - book_genres.csv
    - ratings.csv (full goodbooks5)
    - tags cleaned already
    - produce strict LOO split
    - build 100-negative evaluation data

Outputs saved in new_data_20k/:
    train_df.pkl
    test_df.pkl
    df_dense.pkl
    eval_data.pkl
"""

import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_DIR = "data"  # your goodbooks5 raw directory
GENRE_PATH = "data/book_genres.csv"

SAVE_DIR = "new_data"
os.makedirs(SAVE_DIR, exist_ok=True)


# ------------------------------------------------------------
# Strict leave-one-out split
# ------------------------------------------------------------
def strict_split(df):
    train_rows = []
    test_rows = []

    for uid, grp in df.groupby("user_id"):
        grp_sorted = grp.sort_values("timestamp") if "timestamp" in grp else grp
        if len(grp_sorted) == 1:
            continue
        test = grp_sorted.iloc[-1]
        train = grp_sorted.iloc[:-1]

        train_rows.append(train)
        test_rows.append(test)

    train_df = pd.concat(train_rows)
    test_df = pd.DataFrame(test_rows)

    return train_df, test_df


# ------------------------------------------------------------
# Negative sampling
# ------------------------------------------------------------
def build_eval_data(train_df, test_df, all_items):
    print("\nBuilding 100-negative evaluation sets...")

    train_items = train_df.groupby("user_id")["book_id"].apply(set)
    test_items = test_df.groupby("user_id")["book_id"].apply(set)
    all_users = set(train_items.index).union(test_items.index)

    user_all_items = {
        u: train_items.get(u, set()).union(test_items.get(u, set())) for u in all_users
    }

    eval_data = {}

    for u, grp in tqdm(test_df.groupby("user_id")):
        gt_items = list(grp["book_id"])
        if len(gt_items) == 0:
            continue
        gt = gt_items[0]

        rated = user_all_items[u]
        candidate_pool = list(set(all_items) - rated)

        if len(candidate_pool) < 99:
            continue

        negs = np.random.choice(candidate_pool, 99, replace=False)
        eval_data[u] = {"gt": gt, "neg_items": negs.tolist()}

    return eval_data


# ------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------
def main():
    print("Loading raw data...")

    ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
    genres = pd.read_csv(GENRE_PATH)

    # Keep only books with genre tags
    ratings = ratings[ratings["book_id"].isin(genres["book_id"])]
    print(f"Ratings after removing books without genre tags: {len(ratings)}")

    # --------------------------------------------
    # Filter by book >=30 ratings (global)
    # --------------------------------------------
    book_counts = ratings["book_id"].value_counts()
    good_books = book_counts[book_counts >= 30].index.tolist()

    ratings = ratings[ratings["book_id"].isin(good_books)]
    print(f"Ratings after book>=30 global filter: {len(ratings)}")

    # --------------------------------------------
    # Filter by user ≥30 ratings
    # --------------------------------------------
    user_counts = ratings["user_id"].value_counts()
    good_users = user_counts[user_counts >= 30].index.tolist()

    ratings = ratings[ratings["user_id"].isin(good_users)]
    print(f"Users ≥30 ratings: {len(good_users)}")
    print(f"Ratings after user≥30 filter: {len(ratings)}")

    # --------------------------------------------
    # Select TOP 20,000 most active users
    # --------------------------------------------
    top20k_users = user_counts.sort_values(ascending=False).head(20000).index.tolist()

    ratings_subset = ratings[ratings["user_id"].isin(top20k_users)]
    print(f"\n===== 20,000 USER SUBSET =====")
    print("Users:", len(top20k_users))
    print("Ratings:", len(ratings_subset))
    print("Books:", ratings_subset["book_id"].nunique())

    # --------------------------------------------
    # Ensure books≥30 ratings inside subset
    # --------------------------------------------
    sub_book_counts = ratings_subset["book_id"].value_counts()
    dense_books = sub_book_counts[sub_book_counts >= 30].index.tolist()

    df_dense = ratings_subset[ratings_subset["book_id"].isin(dense_books)]
    print("\n===== FINAL DENSE SET =====")
    print("Users:", df_dense["user_id"].nunique())
    print("Books:", df_dense["book_id"].nunique())
    print("Ratings:", len(df_dense))

    # --------------------------------------------
    # Strict LOO Split
    # --------------------------------------------
    train_df, test_df = strict_split(df_dense)
    print("\nTrain/Test sizes:")
    print("Train:", len(train_df))
    print("Test:", len(test_df))

    # --------------------------------------------
    # Build 100-negative eval sets
    # --------------------------------------------
    all_items = sorted(df_dense["book_id"].unique())
    eval_data = build_eval_data(train_df, test_df, all_items)

    # --------------------------------------------
    # SAVE EVERYTHING
    # --------------------------------------------
    print("\nSaving outputs...")

    df_dense.to_pickle(os.path.join(SAVE_DIR, "df_dense.pkl"))
    train_df.to_pickle(os.path.join(SAVE_DIR, "train_df.pkl"))
    test_df.to_pickle(os.path.join(SAVE_DIR, "test_df.pkl"))

    with open(os.path.join(SAVE_DIR, "eval_data.pkl"), "wb") as f:
        pickle.dump(eval_data, f)

    print("\nDone! Dataset (20k users) successfully created.")


# ------------------------------------------------------------
if __name__ == "__main__":
    main()

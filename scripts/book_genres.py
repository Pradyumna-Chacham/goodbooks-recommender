"""
generate_book_genres_filtered.py

Creates a filtered book_genres file that includes only books appearing
in df_dense.pkl (i.e., after all rating and user/book filtering).

Prevents mismatches during hybrid SVD + RL training.
"""

import pandas as pd

DENSE_PATH = "new_data/df_dense.pkl"
GENRE_PATH = "data/book_genres.csv"
OUT_PATH = "new_data/book_genres_filtered.csv"

print("Loading df_dense...")
df_dense = pd.read_pickle(DENSE_PATH)
dense_books = set(df_dense["book_id"].unique())
print(f"Books in df_dense: {len(dense_books)}")

print("Loading book_genres.csv...")
bg = pd.read_csv(GENRE_PATH)
print(f"Original genre entries: {bg.shape[0]}, Books: {bg['book_id'].nunique()}")

print("Filtering to intersection with df_dense...")
bg_filtered = bg[bg["book_id"].isin(dense_books)]
print(f"Filtered genre entries: {bg_filtered.shape[0]}, Books: {bg_filtered['book_id'].nunique()}")

print(f"Saving filtered genres → {OUT_PATH}")
bg_filtered.to_csv(OUT_PATH, index=False)

print("Done!")

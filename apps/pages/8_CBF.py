import numpy as np
import streamlit as st
from rapidfuzz import fuzz, process
from sklearn.metrics.pairwise import cosine_similarity

from goodbooks_rec.page_common import get_mm_artifacts, get_ui_artifacts
from goodbooks_rec.ui import render_book_recommendations

st.title("🏷️ Content-Based: Similar by Title/Author")

ui = get_ui_artifacts()
A = get_mm_artifacts()

books_df = ui.books[["book_id", "title"]].copy()
books_df["book_id"] = books_df["book_id"].astype(int)
titles = books_df["title"].astype(str).tolist()
title_to_id = dict(zip(books_df["title"].astype(str), books_df["book_id"]))

st.write("Pick 1–3 seed books (type a title; we’ll fuzzy-match).")

seed1 = st.text_input("Seed book 1", value="")
seed2 = st.text_input("Seed book 2 (optional)", value="")
seed3 = st.text_input("Seed book 3 (optional)", value="")

k = st.slider("How many recommendations?", 5, 30, 10)


def match_title(q: str):
    q = q.strip()
    if not q:
        return None
    m = process.extractOne(q, titles, scorer=fuzz.WRatio)
    if m is None:
        return None
    return m[0]  # matched title


matched = []
for s in [seed1, seed2, seed3]:
    mt = match_title(s)
    if mt:
        matched.append(mt)

matched = list(dict.fromkeys(matched))  # de-dup while preserving order

if not matched:
    st.info("Enter at least one seed title.")
    st.stop()

st.caption("Matched seeds: " + ", ".join([f"**{t}**" for t in matched]))

seed_book_ids = [int(title_to_id[t]) for t in matched]

# Convert seed ids to rows in CBF matrix
rows = [A["cbf_book_to_row"].get(b) for b in seed_book_ids]
rows = [r for r in rows if r is not None]

if not rows:
    st.warning("Seed books not found in CBF index.")
    st.stop()

# Mean seed vector; avoid np.matrix issues
user_vec = A["cbf_matrix"][rows].mean(axis=0)
user_vec = np.asarray(user_vec)
if user_vec.ndim == 1:
    user_vec = user_vec.reshape(1, -1)

sims = cosine_similarity(user_vec, A["cbf_matrix"]).ravel()
order = np.argsort(sims)[::-1]

seen = set(seed_book_ids)
out = []
for j in order:
    b = int(A["cbf_book_ids"][int(j)])
    if b in seen:
        continue
    out.append(b)
    if len(out) >= int(k):
        break

render_book_recommendations(
    book_ids=out,
    artifacts=ui,
    page_key="cbf_seeded",
    show_interaction_count=False,
)

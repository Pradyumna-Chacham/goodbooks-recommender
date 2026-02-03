import streamlit as st
from rapidfuzz import fuzz, process

from goodbooks_rec.page_common import get_mm_artifacts, get_ui_artifacts
from goodbooks_rec.ui import render_book_recommendations

st.title("🧩 Item-CF: Similar Books")

ui = get_ui_artifacts()
A = get_mm_artifacts()

books_df = ui.books[["book_id", "title"]].copy()
books_df["book_id"] = books_df["book_id"].astype(int)
titles = books_df["title"].astype(str).tolist()
title_to_id = dict(zip(books_df["title"].astype(str), books_df["book_id"]))

query = st.text_input("Enter a book title", value="")
k = st.slider("How many similar books?", 5, 30, 10)

if not query.strip():
    st.info("Type a title to get Item-CF similar books.")
    st.stop()

match = process.extractOne(query, titles, scorer=fuzz.WRatio)
if match is None:
    st.warning("No close match found.")
    st.stop()

matched_title, score, _ = match
st.caption(f"Closest match: **{matched_title}** (match score: {score})")

seed_book_id = int(title_to_id[matched_title])

book_to_idx = A["book_to_idx"]
idx_to_book = A["idx_to_book"]

if seed_book_id not in book_to_idx:
    st.warning("This book isn't in the model index.")
    st.stop()

seed_idx = book_to_idx[seed_book_id]
nbr_idxs = A["item_topk_idx"][seed_idx]

nbr_book_ids = []
for nidx in nbr_idxs:
    bid = idx_to_book.get(int(nidx))
    if bid is None:
        continue
    bid = int(bid)
    if bid != seed_book_id:
        nbr_book_ids.append(bid)
    if len(nbr_book_ids) >= int(k):
        break

render_book_recommendations(
    book_ids=nbr_book_ids,
    artifacts=ui,
    page_key="item_similar",
    show_interaction_count=False,
)

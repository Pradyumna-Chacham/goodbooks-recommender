import streamlit as st

from goodbooks_rec.io import load_artifacts
from goodbooks_rec.recommend import recommend_popular
from goodbooks_rec.ui import render_book_recommendations

st.title("Popular Recommendations")

k = st.slider("How many Recommendations?", 5, 50, 10)


artifacts = load_artifacts()
recs = recommend_popular(artifacts, k=k)
book_ids = recs["book_id"].astype(int).tolist()

render_book_recommendations(
    book_ids=book_ids, artifacts=artifacts, page_key="popular", show_interaction_count=True
)

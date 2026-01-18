import streamlit as st

from goodbooks_rec.io import load_artifacts
from goodbooks_rec.recommend import recommend_popular

st.set_page_config(page_title="Goodbooks Recommender",page_icon="📚")

st.title("📚 Goodbooks Recommender (MVP)")

k=st.slider("How many recommendations?",5,50,10)


try:
    artifacts=load_artifacts()
    recs=recommend_popular(artifacts,k=k)
    st.dataframe(recs[["book_id","title","n_ratings"]],width="stretch")
except FileNotFoundError as e:
    st.error(str(e))
    st.info("Add the dataset under ./data (books.csv,ratings.csv) or run in demo mode later.")
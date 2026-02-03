import streamlit as st

from goodbooks_rec.demo_users import sample_random_user
from goodbooks_rec.multimodel import recommend_for_user
from goodbooks_rec.page_common import get_mm_artifacts, get_ui_artifacts
from goodbooks_rec.personas import PERSONAS
from goodbooks_rec.ui import render_book_recommendations

st.title("🕹️ RL Reranker (User-based)")

ui = get_ui_artifacts()
A = get_mm_artifacts()

if "rl_user_id" not in st.session_state:
    st.session_state.rl_user_id = sample_random_user(A)

persona = st.selectbox("Persona", list(PERSONAS.keys()), index=0)

c1, c2, c3 = st.columns([3, 1, 1])
with c1:
    user_id = st.number_input(
        "User ID", min_value=1, value=int(st.session_state.rl_user_id), step=1
    )
with c2:
    if st.button("Apply persona", use_container_width=True):
        chosen = PERSONAS.get(persona)
        st.session_state.rl_user_id = sample_random_user(A) if chosen is None else int(chosen)
        st.rerun()
with c3:
    if st.button("🎲 Random", use_container_width=True):
        st.session_state.rl_user_id = sample_random_user(A)
        st.rerun()

st.session_state.rl_user_id = int(user_id)

k = st.slider("How many recommendations?", 5, 50, 10)

book_ids = recommend_for_user("RL", user_id=int(user_id), k=int(k), A=A)
render_book_recommendations(book_ids, ui, page_key="rl_user", show_interaction_count=True)

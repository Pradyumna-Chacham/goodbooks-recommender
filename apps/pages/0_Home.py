import streamlit as st

st.set_page_config(
    page_title="Goodbooks Recommender",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Goodbooks Recommender")

st.markdown("""
This project demonstrates multiple recommendation systems trained on the **Goodbooks dataset**.

Use the links below to explore different models.
""")

st.divider()

st.subheader("🚀 Explore the App")

st.page_link("pages/9_Compare.py", label="Compare All Models", icon="📊")
st.page_link("pages/3_SVD.py", label="SVD (Matrix Factorization)", icon="🧮")
st.page_link("pages/4_User_CF.py", label="User-Based Collaborative Filtering", icon="👥")
st.page_link("pages/5_Item_CF.py", label="Item-Based CF (Similar Books)", icon="🧩")
st.page_link("pages/6_RL.py", label="Reinforcement Learning Reranker", icon="🕹️")
st.page_link("pages/7_Hybrid.py", label="Hybrid (CF + RL)", icon="🔀")
st.page_link("pages/8_CBF.py", label="Content-Based Filtering", icon="🏷️")

st.divider()

st.subheader("ℹ️ About")
st.page_link("pages/99_About.py", label="About the Project", icon="📖")

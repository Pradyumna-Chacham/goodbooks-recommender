import streamlit as st

# Hide sidebar completely on the router page
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Immediately redirect to Landing page
st.switch_page("pages/0_Home.py")

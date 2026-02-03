from pathlib import Path

import streamlit as st

st.set_page_config(page_title="About the Project", page_icon="📖", layout="wide")

st.title("📖 About the Project")

md_path = Path("docs/about.md")

if not md_path.exists():
    st.error("docs/about.md not found")
else:
    st.markdown(md_path.read_text(), unsafe_allow_html=True)

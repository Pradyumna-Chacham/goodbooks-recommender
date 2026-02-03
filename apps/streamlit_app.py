import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
    


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

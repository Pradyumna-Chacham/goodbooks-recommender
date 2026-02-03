import streamlit as st

from goodbooks_rec.io import load_artifacts
from goodbooks_rec.multimodel import load_mm_artifacts


@st.cache_resource
def get_ui_artifacts():
    return load_artifacts()


@st.cache_resource
def get_mm_artifacts():
    return load_mm_artifacts()

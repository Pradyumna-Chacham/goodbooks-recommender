# tests/conftest.py
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures" / "mini"


@pytest.fixture(scope="session")
def mini_meta(fixtures_dir):
    return pd.read_csv(fixtures_dir / "book_metadata.csv")


@pytest.fixture(scope="session")
def mini_svd(fixtures_dir):
    with open(fixtures_dir / "svd_hybrid.pkl", "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="session")
def mini_item_topk(fixtures_dir):
    npz = np.load(fixtures_dir / "item_topk_k10.npz")
    return npz["idx"], npz["sim"]


@pytest.fixture(scope="session")
def mini_user_topk(fixtures_dir):
    npz = np.load(fixtures_dir / "user_topk_k10.npz")
    return npz["idx"], npz["sim"]

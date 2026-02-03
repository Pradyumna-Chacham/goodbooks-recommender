from __future__ import annotations

from pathlib import Path

import pandas as pd

from goodbooks_rec.io import Artifacts

FIXTURES = Path(__file__).parent.parent / "fixtures" / "mini"


def load_mini_artifacts() -> Artifacts:
    """
    Load mini fixtures into the project's Artifacts type
    so tests exercise the real public API.
    """
    books = pd.read_csv(FIXTURES / "book_metadata.csv")
    books["book_id"] = books["book_id"].astype(int)

    ratings = pd.read_pickle(FIXTURES / "train_df.pkl")
    ratings["book_id"] = ratings["book_id"].astype(int)

    # If ratings has columns you don't need, that's fine.
    # recommend_popular only needs "book_id".
    return Artifacts(books=books, ratings=ratings)

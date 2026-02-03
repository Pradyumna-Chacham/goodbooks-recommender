import pandas as pd

from goodbooks_rec.recommend import recommend_popular
from tests.helpers.load_mini_artifacts import load_mini_artifacts


def test_recommend_popular_returns_topk_sorted():
    artifacts = load_mini_artifacts()  # this will now return an Artifacts object (patch 2)
    k = 10

    df = recommend_popular(artifacts, k=k)

    assert isinstance(df, pd.DataFrame)
    assert "book_id" in df.columns
    assert "n_ratings" in df.columns
    assert len(df) <= k

    # sorted descending by n_ratings
    assert df["n_ratings"].is_monotonic_decreasing

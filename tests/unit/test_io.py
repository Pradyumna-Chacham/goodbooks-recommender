import pickle
from pathlib import Path

import pandas as pd

from goodbooks_rec.io import load_artifacts


def test_load_artifacts_uses_demo_when_real_data_missing(tmp_path: Path):
    # tmp_path has no books.csv/ratings.csv -> should fall back to demo
    artifacts = load_artifacts(data_dir=tmp_path, models_dir=tmp_path)

    assert artifacts.demo_mode is True
    assert isinstance(artifacts.books, pd.DataFrame)
    assert isinstance(artifacts.ratings, pd.DataFrame)
    assert len(artifacts.books) > 0
    assert len(artifacts.ratings) > 0


def test_load_artifacts_uses_real_data_when_present(tmp_path: Path):
    books_path = tmp_path / "books.csv"
    ratings_path = tmp_path / "ratings.csv"

    pd.DataFrame(
        {
            "book_id": [1, 2],
            "title": ["A", "B"],
            "authors": ["x", "y"],
        }
    ).to_csv(books_path, index=False)

    pd.DataFrame(
        {
            "user_id": [10, 11, 10],
            "book_id": [1, 1, 2],
            "rating": [5, 4, 3],
        }
    ).to_csv(ratings_path, index=False)

    artifacts = load_artifacts(data_dir=tmp_path, models_dir=tmp_path)

    assert artifacts.demo_mode is False
    assert artifacts.books["book_id"].tolist() == [1, 2]
    assert set(artifacts.ratings.columns) >= {"user_id", "book_id", "rating"}


def test_load_artifacts_demo_mode_requires_both_real_files(tmp_path: Path):
    # create only books.csv, but no ratings.csv
    pd.DataFrame({"book_id": [1], "title": ["A"], "authors": ["x"]}).to_csv(
        tmp_path / "books.csv", index=False
    )

    artifacts = load_artifacts(data_dir=tmp_path, models_dir=tmp_path)

    assert artifacts.demo_mode is True
    assert len(artifacts.books) > 0
    assert len(artifacts.ratings) > 0


def test_load_artifacts_loads_popularity_model_when_present(tmp_path: Path):
    # Provide real data so it doesn't use demo (not strictly required, but makes test fully isolated)
    pd.DataFrame(
        {
            "book_id": [1],
            "title": ["A"],
            "authors": ["x"],
        }
    ).to_csv(tmp_path / "books.csv", index=False)

    pd.DataFrame(
        {
            "user_id": [10],
            "book_id": [1],
            "rating": [5],
        }
    ).to_csv(tmp_path / "ratings.csv", index=False)

    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    payload = {"hello": "world", "top": [1, 2, 3]}
    with open(models_dir / "popularity.pkl", "wb") as f:
        pickle.dump(payload, f)

    artifacts = load_artifacts(data_dir=tmp_path, models_dir=models_dir)

    assert artifacts.demo_mode is False
    assert artifacts.model_popularity == payload


def test_load_artifacts_model_popularity_none_when_missing(tmp_path: Path):
    pd.DataFrame(
        {
            "book_id": [1],
            "title": ["A"],
            "authors": ["x"],
        }
    ).to_csv(tmp_path / "books.csv", index=False)

    pd.DataFrame(
        {
            "user_id": [10],
            "book_id": [1],
            "rating": [5],
        }
    ).to_csv(tmp_path / "ratings.csv", index=False)

    artifacts = load_artifacts(data_dir=tmp_path, models_dir=tmp_path)

    assert artifacts.demo_mode is False
    assert artifacts.model_popularity is None

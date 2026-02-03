from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import DATA_DIR, MODELS_DIR


@dataclass(frozen=True)
class Artifacts:
    books: pd.DataFrame
    ratings: pd.DataFrame
    model_popularity: object | None = None
    demo_mode: bool = False


def load_artifacts(data_dir: Path = DATA_DIR, models_dir=MODELS_DIR) -> Artifacts:
    # ---- define real paths ----
    real_books = data_dir / "books.csv"
    real_ratings = data_dir / "ratings.csv"

    # ---- define demo paths ----
    demo_dir = Path(__file__).resolve().parent / "demo"
    demo_books = demo_dir / "books.csv"
    demo_ratings = demo_dir / "ratings.csv"

    # ---- decide which to use ----
    use_demo = not (real_books.exists() and real_ratings.exists())

    books_path = demo_books if use_demo else real_books
    ratings_path = demo_ratings if use_demo else real_ratings

    # ✅ ---- THESE ARE THE READS ----
    books = pd.read_csv(books_path)
    ratings = pd.read_csv(ratings_path)

    # ---- optional model ----
    model_popularity = None
    pop_path = models_dir / "popularity.pkl"
    if pop_path.exists():
        with open(pop_path, "rb") as f:
            model_popularity = pickle.load(f)

    return Artifacts(
        books=books,
        ratings=ratings,
        model_popularity=model_popularity,
        demo_mode=use_demo,
    )

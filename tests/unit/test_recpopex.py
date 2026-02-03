import pandas as pd

from goodbooks_rec.io import Artifacts
from goodbooks_rec.recommend import recommend_popular


def test_recommend_popular_exact_counts():
    books = pd.DataFrame(
        {
            "book_id": [1, 2, 3],
            "title": ["A", "B", "C"],
            "authors": ["x", "y", "z"],
        }
    )
    ratings = pd.DataFrame(
        {
            "user_id": [10, 11, 12, 10, 11],
            "book_id": [1, 1, 2, 1, 3],
            "rating": [5, 4, 5, 3, 2],
        }
    )
    artifacts = Artifacts(books=books, ratings=ratings)

    df = recommend_popular(artifacts, k=3)

    # book 1 appears 3 times, book 2 once, book 3 once
    assert df.loc[df["book_id"] == 1, "n_ratings"].iloc[0] == 3
    # top row should be book_id 1
    assert df.iloc[0]["book_id"] == 1


def test_recommend_popular_missing_book_metadata():
    books = pd.DataFrame({"book_id": [1], "title": ["A"], "authors": ["x"]})
    ratings = pd.DataFrame({"user_id": [1, 2, 3], "book_id": [999, 999, 1], "rating": [5, 4, 3]})
    artifacts = Artifacts(books=books, ratings=ratings)

    df = recommend_popular(artifacts, k=2)

    assert 999 in df["book_id"].tolist()


def test_recommend_popular_k_larger_than_books():
    books = pd.DataFrame({"book_id": [1, 2], "title": ["A", "B"], "authors": ["x", "y"]})
    ratings = pd.DataFrame({"user_id": [1, 2], "book_id": [1, 2], "rating": [5, 4]})
    artifacts = Artifacts(books=books, ratings=ratings)

    df = recommend_popular(artifacts, k=10)
    assert len(df) == 2

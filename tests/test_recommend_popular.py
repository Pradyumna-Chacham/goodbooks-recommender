import pandas as pd
from goodbooks_rec.recommend import recommend_popular
from goodbooks_rec.io import Artifacts


def test_recommend_popular_basic():
    books = pd.DataFrame({
        "book_id": [1, 2, 3],
        "title": ["A", "B", "C"],
    })

    ratings = pd.DataFrame({
        "user_id": [1, 2, 3, 1],
        "book_id": [1, 1, 2, 1],
    })

    artifacts = Artifacts(books=books, ratings=ratings)

    recs = recommend_popular(artifacts, k=1)

    assert recs.iloc[0]["book_id"] == 1
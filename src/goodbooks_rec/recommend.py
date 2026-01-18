from __future__ import annotations

import pandas as pd

from .io import Artifacts

def recommend_popular(artifacts:Artifacts,k: int=10) -> pd.DataFrame:
    """
    Docstring for recommend_popular
    
    :param artifacts: books,rating and popularity
    :type artifacts: Artifacts
    :param k: No.of books
    :type k: int
    :return: Return top-k most popular books by rating count
    :rtype: DataFrame
    """

    ratings=artifacts.ratings
    books = artifacts.books

    counts=(
        ratings.groupby("book_id").size().reset_index(name="n_ratings").sort_values("n_ratings",ascending=False).head(k)
    )

    return counts.merge(books,on="book_id",how="left")


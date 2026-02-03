# tests/integration/test_cold_start.py
from goodbooks_rec.recommend import recommend_popular
from tests.helpers.load_mini_artifacts import load_mini_artifacts


def test_cold_start_user_does_not_crash():
    A = load_mini_artifacts()

    unknown_user = -999999
    recs = recommend_popular(A, user_id=unknown_user, k=5)

    assert isinstance(recs, list)
    assert len(recs) <= 5

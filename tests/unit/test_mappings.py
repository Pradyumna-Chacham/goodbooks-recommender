# tests/unit/test_mappings.py
def test_id_index_roundtrip(mini_svd):
    user_to_idx = mini_svd["user_to_idx"]
    book_to_idx = mini_svd["book_to_idx"]
    idx_to_user = {v: k for k, v in user_to_idx.items()}
    idx_to_book = {v: k for k, v in book_to_idx.items()}

    for uid, i in list(user_to_idx.items())[:5]:
        assert idx_to_user[i] == uid

    for bid, i in list(book_to_idx.items())[:5]:
        assert idx_to_book[i] == bid

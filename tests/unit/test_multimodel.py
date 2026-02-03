import numpy as np

from goodbooks_rec.multimodel import _load_torch_and_qnet, _zscore_dict


def test_zscore_dict_empty():
    assert _zscore_dict({}) == {}


def test_zscore_dict_standardizes_mean_zero():
    d = {1: 10.0, 2: 20.0, 3: 30.0}
    z = _zscore_dict(d)
    vals = np.array(list(z.values()), dtype=np.float32)
    assert abs(float(vals.mean())) < 1e-6
    assert float(vals.std()) > 0.9  # ~1


from goodbooks_rec.multimodel import recommend_popular_user
from tests.helpers.mini_mm_A import build_mini_A_for_multimodel


def test_recommend_popular_user_filters_seen_and_length():
    A = build_mini_A_for_multimodel(k=10)
    user_id = next(iter(A["user_hist"]))  # has history
    seen = set(A["user_hist"][user_id])

    recs = recommend_popular_user(user_id, k=10, A=A)

    assert len(recs) == len(set(recs))
    assert len(recs) <= 10
    assert not any(b in seen for b in recs)


from goodbooks_rec.multimodel import recommend_popular_user, recommend_user_cf
from tests.helpers.mini_mm_A import build_mini_A_for_multimodel


def test_recommend_user_cf_unknown_user_fallback():
    A = build_mini_A_for_multimodel(k=10)
    unknown_user = -999999

    recs = recommend_user_cf(unknown_user, k=10, A=A)
    pop = recommend_popular_user(unknown_user, k=10, A=A)

    assert recs == pop


from goodbooks_rec.multimodel import recommend_item_cf
from tests.helpers.mini_mm_A import build_mini_A_for_multimodel


def test_recommend_item_cf_returns_unseen():
    A = build_mini_A_for_multimodel(k=10)
    user_id = next(iter(A["user_hist"]))
    seen = set(A["user_hist"][user_id])

    recs = recommend_item_cf(user_id, k=10, A=A)

    assert len(recs) <= 10
    assert len(recs) == len(set(recs))
    assert not any(b in seen for b in recs)


from goodbooks_rec.multimodel import recommend_popular_user, recommend_svd
from tests.helpers.mini_mm_A import build_mini_A_for_multimodel


def test_recommend_svd_unknown_user_fallback():
    A = build_mini_A_for_multimodel(k=10)
    unknown_user = -999999
    recs = recommend_svd(unknown_user, k=10, A=A)
    assert recs == recommend_popular_user(unknown_user, k=10, A=A)


def test_recommend_svd_filters_seen():
    A = build_mini_A_for_multimodel(k=10)
    user_id = next(iter(A["user_hist"]))
    seen = set(A["user_hist"][user_id])

    recs = recommend_svd(user_id, k=10, A=A)

    assert len(recs) <= 10
    assert not any(b in seen for b in recs)


from goodbooks_rec.multimodel import recommend_cbf
from tests.helpers.mini_mm_A import build_mini_A_for_multimodel


def test_recommend_cbf_runs_and_filters_seen():
    A = build_mini_A_for_multimodel(k=10)
    user_id = next(iter(A["user_hist"]))
    seen = set(A["user_hist"][user_id])

    recs = recommend_cbf(user_id, k=10, A=A)

    assert len(recs) <= 10
    assert len(recs) == len(set(recs))
    assert not any(b in seen for b in recs)


import numpy as np

import goodbooks_rec.multimodel as mm
from tests.helpers.mini_mm_A import build_mini_A_for_multimodel


def test_recommend_hybrid_runs_without_torch(monkeypatch):
    A = build_mini_A_for_multimodel(k=10)
    user_id = next(iter(A["user_hist"]))

    def fake_score_with_rl(user_id, candidates, A):
        # deterministic and fast
        return np.zeros(len(candidates), dtype=np.float32)

    monkeypatch.setattr(mm, "_score_with_rl", fake_score_with_rl)

    recs = mm.recommend_hybrid(user_id, k=10, A=A)

    assert len(recs) <= 10
    assert len(recs) == len(set(recs))


import goodbooks_rec.multimodel as mm
from tests.helpers.mini_mm_A import build_mini_A_for_multimodel


def test_recommend_for_user_router_defaults_to_popular():
    A = build_mini_A_for_multimodel(k=10)
    user_id = next(iter(A["user_hist"]))

    recs_unknown = mm.recommend_for_user("NOT_A_MODEL", user_id, 5, A)
    recs_pop = mm.recommend_for_user("Popular", user_id, 5, A)

    assert recs_unknown == recs_pop


def test_recommend_for_user_router_all_models_run(monkeypatch):
    A = build_mini_A_for_multimodel(k=10)
    user_id = next(iter(A["user_hist"]))

    # patch RL scoring so RL + Hybrid don’t need torch
    import numpy as np

    monkeypatch.setattr(mm, "_score_with_rl", lambda u, c, A: np.zeros(len(c), dtype=np.float32))

    for name in ["Popular", "User-CF", "Item-CF", "SVD", "RL", "Hybrid", "Content-Based"]:
        recs = mm.recommend_for_user(name, user_id, 5, A)
        assert isinstance(recs, list)
        assert len(recs) <= 5


def test_qnet_forward_shape():
    torch, QNet = _load_torch_and_qnet()
    qnet = QNet(dim=150)
    x = torch.zeros((4, 150), dtype=torch.float32)
    y = qnet(x)
    assert tuple(y.shape) == (4, 1)


def test_score_with_rl_returns_array(monkeypatch):
    A = build_mini_A_for_multimodel(k=10)

    # create a minimal torch + qnet from the real helper
    torch, QNet = mm._load_torch_and_qnet()
    device = torch.device("cpu")

    # build fake tensors consistent with embeddings
    user_embed_np = A["user_embed_np"]
    item_embed_np = A["item_embed_np"]

    A["torch"] = torch
    A["device"] = device
    A["qnet"] = QNet(dim=150).to(device).eval()
    A["user_embed_t"] = torch.tensor(user_embed_np, dtype=torch.float32)
    A["item_embed_t"] = torch.tensor(item_embed_np, dtype=torch.float32)
    # cf_vecs_t: use zeros (shape n_items,50)
    A["cf_vecs_t"] = torch.zeros((item_embed_np.shape[0], 50), dtype=torch.float32)

    user_id = next(iter(A["user_to_idx"].keys()))
    candidates = A["pop_sorted"][:20]

    scores = mm._score_with_rl(user_id, candidates, A)

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (len(candidates),)
    assert scores.dtype == np.float32

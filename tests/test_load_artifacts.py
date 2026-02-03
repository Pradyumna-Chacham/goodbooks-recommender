# tests/test_load_artifacts.py
from pathlib import Path

import pytest

from goodbooks_rec.multimodel import \
    load_mm_artifacts  # adjust import to your real location


def test_load_artifacts_missing_data(tmp_path: Path):
    missing_item = tmp_path / "missing_item_topk.npz"
    missing_user = tmp_path / "missing_user_topk.npz"

    with pytest.raises(FileNotFoundError):
        load_mm_artifacts(item_topk_path=missing_item, user_topk_path=missing_user)

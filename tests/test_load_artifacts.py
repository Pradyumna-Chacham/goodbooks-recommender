import pytest 
from pathlib import Path
from goodbooks_rec.io import load_artifacts

def test_load_artifacts_missing_data(tmp_path:Path):
    with pytest.raises(FileNotFoundError) as e:
        load_artifacts(data_dir=tmp_path,models_dir=tmp_path)
    assert "books.csv" in str(e.value)
    assert "ratings.csv" in str(e.value)

    
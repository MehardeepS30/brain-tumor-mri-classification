"""Download helpers: credential discovery and post-download verification.

The real download is exercised by scripts/download_data.py, not here.
"""

import pytest
from PIL import Image

from brain_tumor.download import find_kaggle_credentials, scan_raw_dataset


def test_finds_legacy_kaggle_json(tmp_path):
    (tmp_path / ".kaggle").mkdir()
    (tmp_path / ".kaggle" / "kaggle.json").write_text("{}")
    assert find_kaggle_credentials(home=tmp_path, env={}) == "kaggle.json"


def test_finds_new_style_token_in_env(tmp_path):
    assert find_kaggle_credentials(home=tmp_path, env={"KAGGLE_API_TOKEN": "x"}) == "KAGGLE_API_TOKEN"


def test_finds_legacy_env_pair(tmp_path):
    env = {"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k"}
    assert find_kaggle_credentials(home=tmp_path, env=env) == "KAGGLE_USERNAME/KAGGLE_KEY"


def test_missing_credentials_raise_with_instructions(tmp_path):
    with pytest.raises(FileNotFoundError, match="kaggle.json"):
        find_kaggle_credentials(home=tmp_path, env={})


def _make_raw(root):
    for part in ("Training", "Testing"):
        for cls in ("glioma", "meningioma", "notumor", "pituitary"):
            d = root / part / cls
            d.mkdir(parents=True)
            Image.new("L", (32, 32), 100).save(d / f"{cls}_1.jpg")
    return root


def test_scan_lists_every_image_with_label_and_source_folder(tmp_path):
    rows, problems = scan_raw_dataset(_make_raw(tmp_path))
    assert len(rows) == 8 and not problems
    assert {r["label"] for r in rows} == {"glioma", "meningioma", "notumor", "pituitary"}
    assert {r["source_split"] for r in rows} == {"Training", "Testing"}
    assert all(r["path"].count("/") == 2 for r in rows)  # relative, forward slashes


def test_scan_reports_unreadable_files_and_unknown_folders(tmp_path):
    root = _make_raw(tmp_path)
    (root / "Training" / "glioma" / "broken.jpg").write_bytes(b"not an image")
    (root / "Training" / "tumour").mkdir()
    Image.new("L", (8, 8)).save(root / "Training" / "tumour" / "x.jpg")
    rows, problems = scan_raw_dataset(root)
    assert len(rows) == 8
    kinds = sorted(p["problem"] for p in problems)
    assert kinds == ["unknown label folder", "unreadable image"]

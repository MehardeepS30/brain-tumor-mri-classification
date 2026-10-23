"""Kaggle credential discovery, dataset download, and verification of the raw files."""

import os
from pathlib import Path

from PIL import Image

from brain_tumor.classes import CLASS_NAMES

DATASET = "masoudnickparvar/brain-tumor-mri-dataset"
SOURCE_SPLITS = ("Training", "Testing")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def find_kaggle_credentials(home=None, env=None):
    """Name the credential the Kaggle CLI will use, or raise with setup instructions.

    Accepts the new-style API token (env var or ~/.kaggle/access_token), the legacy
    KAGGLE_USERNAME/KAGGLE_KEY pair, or a legacy ~/.kaggle/kaggle.json."""
    home = Path.home() if home is None else Path(home)
    env = os.environ if env is None else env
    if env.get("KAGGLE_API_TOKEN"):
        return "KAGGLE_API_TOKEN"
    if env.get("KAGGLE_USERNAME") and env.get("KAGGLE_KEY"):
        return "KAGGLE_USERNAME/KAGGLE_KEY"
    if (home / ".kaggle" / "access_token").is_file():
        return "access_token"
    if (home / ".kaggle" / "kaggle.json").is_file():
        return "kaggle.json"
    raise FileNotFoundError(
        f"No Kaggle credentials found. On kaggle.com > Settings > API, click "
        f"'Create Legacy API Key' and save the downloaded kaggle.json to "
        f"{home / '.kaggle' / 'kaggle.json'}, or set KAGGLE_API_TOKEN."
    )


def download_dataset(dest):
    """Download and unzip the dataset into `dest` with the Kaggle API."""
    find_kaggle_credentials()
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    Path(dest).mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(DATASET, path=str(dest), unzip=True, quiet=False)


def scan_raw_dataset(root):
    """List every image under root/{Training,Testing}/<class>/ after fully decoding it.

    Returns (rows, problems). rows: dicts with path (relative, forward slashes),
    label, source_split, width, height. problems: dicts with path and problem."""
    root = Path(root)
    rows, problems = [], []
    for part in SOURCE_SPLITS:
        part_dir = root / part
        if not part_dir.is_dir():
            problems.append({"path": part, "problem": "missing split folder"})
            continue
        for cls_dir in sorted(p for p in part_dir.iterdir() if p.is_dir()):
            rel_dir = f"{part}/{cls_dir.name}"
            if cls_dir.name not in CLASS_NAMES:
                problems.append({"path": rel_dir, "problem": "unknown label folder"})
                continue
            for f in sorted(cls_dir.iterdir()):
                if f.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                rel = f"{rel_dir}/{f.name}"
                try:
                    with Image.open(f) as img:
                        img.load()
                        w, h = img.size
                except Exception:
                    problems.append({"path": rel, "problem": "unreadable image"})
                    continue
                rows.append({"path": rel, "label": cls_dir.name, "source_split": part,
                             "width": w, "height": h})
    return rows, problems

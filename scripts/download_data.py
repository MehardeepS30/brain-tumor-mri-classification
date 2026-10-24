"""Step 0: download the Kaggle dataset zip, record its SHA-256, extract to data/raw.

Needs Kaggle credentials (~/.kaggle/kaggle.json, ~/.kaggle/access_token or
KAGGLE_API_TOKEN). Skips the download when the zip is already present, and skips
extraction when data/raw already exists. Extracted files are marked read-only.
"""

import argparse
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

from brain_tumor.dedupe import sha256_file
from brain_tumor.download import DATASET, find_kaggle_credentials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip-dir", default="data/zip")
    ap.add_argument("--raw", default="data/raw")
    args = ap.parse_args()
    zip_dir, raw = Path(args.zip_dir), Path(args.raw)
    zpath = zip_dir / (DATASET.split("/")[1] + ".zip")

    if not zpath.exists():
        print("credentials:", find_kaggle_credentials())
        zip_dir.mkdir(parents=True, exist_ok=True)
        kaggle = Path(sys.executable).with_name("kaggle")
        subprocess.run([str(kaggle), "datasets", "download", DATASET, "-p", str(zip_dir)], check=True)
    digest = sha256_file(zpath)
    (zip_dir / "SHA256SUMS").write_text(f"{digest}  {zpath.name}\n")
    print(f"{zpath} sha256 {digest}")

    if raw.exists():
        print(f"{raw} exists, not extracting")
        return
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        if any(n.startswith("/") or ".." in Path(n).parts for n in names):
            raise ValueError("zip contains unsafe paths")
        z.extractall(raw)
    for f in raw.rglob("*"):
        if f.is_file():
            f.chmod(stat.S_IREAD)
    print(f"extracted {sum(1 for n in names if not n.endswith('/'))} files to {raw} (read-only)")


if __name__ == "__main__":
    main()

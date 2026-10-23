"""Step 1 of the data stage: inventory data/raw without modifying it.

Decodes every image, records size/mode/metadata, SHA-256 and 64-bit pHash, and
writes data/interim/images.csv plus data/interim/scan_problems.csv.
"""

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from brain_tumor.dedupe import phash64, sha256_file
from brain_tumor.download import scan_raw_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/interim")
    args = ap.parse_args()
    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, problems = scan_raw_dataset(raw)
    for r in tqdm(rows, desc="hashing"):
        f = raw / r["path"]
        with Image.open(f) as img:
            r["mode"] = img.mode
            r["format"] = img.format
            r["info_keys"] = ";".join(sorted(k for k in img.info if k not in ("jfif", "jfif_version", "jfif_unit", "jfif_density", "dpi", "progressive", "progression")))
            r["exif_tags"] = len(img.getexif())
            r["phash"] = f"{phash64(img):016x}"
        r["sha256"] = sha256_file(f)
        r["bytes"] = f.stat().st_size

    pd.DataFrame(rows).to_csv(out / "images.csv", index=False)
    pd.DataFrame(problems, columns=["path", "problem"]).to_csv(out / "scan_problems.csv", index=False)
    print(f"{len(rows)} images, {len(problems)} problems -> {out}")


if __name__ == "__main__":
    main()

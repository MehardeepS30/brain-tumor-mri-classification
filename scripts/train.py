"""Train one model from a JSON config: python scripts/train.py configs/baseline.json [--resume]

Any config field can be overridden on the command line, e.g. --epochs 2 --out_dir outputs/x.
"""

import argparse
import json
from dataclasses import fields
from pathlib import Path

import pandas as pd

from brain_tumor.train import TrainConfig, fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--cache", default="data/processed")
    known, rest = ap.parse_known_args()
    cfg = json.loads(Path(known.config).read_text())
    types = {f.name: f.type for f in fields(TrainConfig)}
    it = iter(rest)
    for flag in it:
        key = flag.lstrip("-")
        if key == "resume":
            cfg["resume"] = True
            continue
        val = next(it)
        cfg[key] = json.loads(val) if types.get(key) not in (str, "str") else val
    best = fit(TrainConfig(**cfg), pd.read_csv(known.manifest), known.cache)
    print(f"best val macro recall: {best:.4f}")


if __name__ == "__main__":
    main()

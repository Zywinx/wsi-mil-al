import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile_manifest", type=str, default="data/metadata/tile_manifest.csv")
    ap.add_argument("--out_json", type=str, default="data/metadata/bag_index.json")
    args = ap.parse_args()

    df = pd.read_csv(args.tile_manifest)
    bags = defaultdict(list)
    for _, r in df.iterrows():
        bags[r["slide_id"]].append(
            dict(
                tile_path=r["tile_path"],
                x=int(r["x"]),
                y=int(r["y"]),
                tissue_ratio=float(r["tissue_ratio"]),
            )
        )

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(bags, f, ensure_ascii=False)
    print("Saved:", out, "n_slides=", len(bags))

if __name__ == "__main__":
    main()
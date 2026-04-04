import argparse
from pathlib import Path

import cv2
import numpy as np
import openslide
import pandas as pd
from PIL import Image
from tqdm import tqdm


def make_tissue_mask(slide, mask_max_dim=4096):
    w, h = slide.dimensions
    scale = max(w, h) / mask_max_dim if max(w, h) > mask_max_dim else 1.0
    thumb_w, thumb_h = max(1, int(w / scale)), max(1, int(h / scale))
    thumb = slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB")
    arr = np.array(thumb)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    # simple tissue heuristic: not white background and has some saturation
    mask = ((sat > 20) & (val < 245)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mask, scale


def tile_tissue_ratio(mask, scale, x, y, tile_size):
    mx0 = int(x / scale)
    my0 = int(y / scale)
    mx1 = int((x + tile_size) / scale)
    my1 = int((y + tile_size) / scale)
    H, W = mask.shape
    mx0 = max(0, min(mx0, W))
    mx1 = max(0, min(mx1, W))
    my0 = max(0, min(my0, H))
    my1 = max(0, min(my1, H))
    if mx1 <= mx0 or my1 <= my0:
        return 0.0
    region = mask[my0:my1, mx0:mx1]
    return float(region.mean())


def find_first_svs(slide_dir: Path):
    svs = list(slide_dir.glob("*.svs"))
    if len(svs) == 0:
        return None
    return svs[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_csv", type=str, default="data/metadata/splits_patient.csv")
    ap.add_argument("--raw_dir", type=str, default="data/raw/tcga_brca_svs")
    ap.add_argument("--tiles_dir", type=str, default="data/tiles")
    ap.add_argument("--manifest_csv", type=str, default="data/metadata/tile_manifest.csv")
    ap.add_argument("--tile_size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--tissue_ratio_thr", type=float, default=0.5)
    ap.add_argument("--mask_max_dim", type=int, default=4096)
    ap.add_argument("--limit_slides", type=int, default=-1)

    # new args for sharding
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_rank", type=int, default=0)

    args = ap.parse_args()

    assert 0 <= args.shard_rank < args.num_shards, "shard_rank must be in [0, num_shards)"

    df = pd.read_csv(args.splits_csv)
    slide_ids = df["slide_id"].astype(str).tolist()

    if args.limit_slides > 0:
        slide_ids = slide_ids[: args.limit_slides]

    # shard by slide index
    slide_ids = slide_ids[args.shard_rank::args.num_shards]

    raw_dir = Path(args.raw_dir)
    tiles_dir = Path(args.tiles_dir)
    tiles_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    for slide_id in tqdm(slide_ids, desc=f"extract tiles shard {args.shard_rank}/{args.num_shards}"):
        slide_dir = raw_dir / slide_id
        svs_path = find_first_svs(slide_dir)
        if svs_path is None:
            print(f"[WARN] slide_id not found in raw_dir: {slide_id}")
            continue

        try:
            slide = openslide.OpenSlide(str(svs_path))
        except Exception as e:
            print(f"[WARN] failed to open {svs_path}: {e}")
            continue

        w, h = slide.dimensions
        mask, scale = make_tissue_mask(slide, mask_max_dim=args.mask_max_dim)

        out_dir = tiles_dir / slide_id
        out_dir.mkdir(parents=True, exist_ok=True)

        tile_idx = 0
        for y in range(0, max(1, h - args.tile_size + 1), args.stride):
            for x in range(0, max(1, w - args.tile_size + 1), args.stride):
                ratio = tile_tissue_ratio(mask, scale, x, y, args.tile_size)
                if ratio < args.tissue_ratio_thr:
                    continue

                try:
                    region = slide.read_region((x, y), 0, (args.tile_size, args.tile_size)).convert("RGB")
                except Exception as e:
                    print(f"[WARN] read_region failed slide={slide_id} x={x} y={y}: {e}")
                    continue

                tile_name = f"{slide_id}_x{x}_y{y}.png"
                tile_path = out_dir / tile_name

                # smaller compression = faster write, same pixels
                region.save(tile_path, compress_level=1)

                manifest_rows.append(
                    dict(
                        slide_id=slide_id,
                        tile_id=tile_idx,
                        x=x,
                        y=y,
                        tile_size=args.tile_size,
                        stride=args.stride,
                        tissue_ratio=ratio,
                        tile_path=str(tile_path),
                    )
                )
                tile_idx += 1

        slide.close()

    manifest_csv = Path(args.manifest_csv)
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest_rows).to_csv(manifest_csv, index=False)
    print("Saved:", manifest_csv, "n_rows=", len(manifest_rows))


if __name__ == "__main__":
    main()

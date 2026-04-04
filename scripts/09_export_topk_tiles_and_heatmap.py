import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm import tqdm

from wsi_mil.utils.io import read_json
from wsi_mil.models.wsi_mil_model import WSIBaselineMIL
from wsi_mil.utils.vis import save_topk_mosaic


def load_tile_tensor(p: str, img_size: int):
    img = Image.open(p).convert("RGB").resize((img_size, img_size))
    arr = np.array(img).astype(np.float32) / 255.0
    # ImageNet normalize
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    arr = (arr - mean) / std
    x = torch.from_numpy(arr).permute(2, 0, 1)
    return x

@torch.no_grad()
def infer_slide_all_tiles(model, tile_records, device, img_size: int, tile_bs: int = 64):
    # compute z for all tiles in chunks
    model.eval()
    n_tiles = len(tile_records)
    z_list = []
    for i in range(0, n_tiles, tile_bs):
        batch_records = tile_records[i:i+tile_bs]
        xs = [load_tile_tensor(tr["tile_path"], img_size) for tr in batch_records]
        X = torch.stack(xs, dim=0).to(device)  # [bs,3,H,W]
        zb = model.encoder(X)  # [bs,D]
        z_list.append(zb.cpu())
        del X, xs  # 及时释放内存
    
    z = torch.cat(z_list, dim=0).unsqueeze(0).to(device)   # [1,N,D]
    slide_logit, alpha, h, e = model.mil(z)
    slide_prob = torch.sigmoid(slide_logit).item()
    alpha = alpha.squeeze(0).detach().cpu().numpy()
    return slide_prob, alpha, e.squeeze(0).detach().cpu().numpy()
    """
    tiles = tile_records
    xs = []
    for tr in tiles:
        xs.append(load_tile_tensor(tr["tile_path"], img_size))
    X = torch.stack(xs, dim=0)  # [N,3,H,W]
    # encode
    z_list = []
    for i in range(0, X.size(0), tile_bs):
        xb = X[i:i+tile_bs].to(device)
        zb = model.encoder(xb)  # [bs,D]
        z_list.append(zb.cpu())
    z = torch.cat(z_list, dim=0)  # [N,D]
    z = z.unsqueeze(0).to(device)  # [1,N,D]
    # MIL
    slide_logit, alpha, h, e = model.mil(z)
    slide_prob = torch.sigmoid(slide_logit).item()
    alpha = alpha.squeeze(0).detach().cpu().numpy()
    return slide_prob, alpha, e.squeeze(0).detach().cpu().numpy()
    """

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/stage1_baseline.yaml")
    ap.add_argument("--ckpt", type=str, default="runs/stage1_baseline/ckpt_best.pt")
    ap.add_argument("--bag_index", type=str, default="data/metadata/bag_index.json")
    ap.add_argument("--splits_csv", type=str, default="data/metadata/splits_patient.csv")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--tile_bs", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    out_dir = Path(cfg["experiment"]["out_dir"])
    topk_dir = out_dir / "topk_tiles"
    heat_dir = out_dir / "heatmaps"
    topk_dir.mkdir(parents=True, exist_ok=True)
    heat_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = WSIBaselineMIL(**cfg["model"]).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)

    bags = read_json(args.bag_index)
    df = pd.read_csv(args.splits_csv)
    df = df[df["split"] == args.split].reset_index(drop=True)

    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="export topk"):
        slide_id = str(r["slide_id"])
        if slide_id not in bags:
            continue
        tile_records = bags[slide_id]
        slide_prob, alpha, _e = infer_slide_all_tiles(model, tile_records, device, cfg["data"]["img_size"], tile_bs=args.tile_bs)

        # top-k by alpha
        idx = np.argsort(-alpha)[: args.topk]
        top_paths = [tile_records[i]["tile_path"] for i in idx]
        top_scores = [float(alpha[i]) for i in idx]

        # save mosaic
        mosaic_png = topk_dir / f"{slide_id}.png"
        save_topk_mosaic(top_paths, top_scores, str(mosaic_png), thumb_size=cfg["data"]["img_size"])

        #绘制热力图
        # save simple attention heatmap
        xs = np.array([int(tr["x"]) for tr in tile_records], dtype=np.int32)
        ys = np.array([int(tr["y"]) for tr in tile_records], dtype=np.int32)

        tile_size = cfg["data"]["tile_size",256]
        if len(xs) > 0:
            x0, y0 = xs.min(), ys.min()
            x1, y1 = xs.max(), ys.max()

            w = int((x1 - x0) / tile_size) + 1
            h = int((y1 - y0) / tile_size) + 1

            heat = np.zeros((h, w), dtype=np.float32)
            count = np.zeros((h, w), dtype=np.float32)

            for tr, a in zip(tile_records, alpha):
                gx = int((int(tr["x"]) - x0) / tile_size)
                gy = int((int(tr["y"]) - y0) / tile_size)
                heat[gy, gx] += float(a)
                count[gy, gx] += 1.0

            heat = heat / np.maximum(count, 1e-6)
            heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
            """
            heat_img = (heat * 255).astype(np.uint8)

            heat_png = heat_dir / f"{slide_id}.png"
            Image.fromarray(heat_img).save(heat_png)
            """

            heat_png = heat_dir / f"{slide_id}.png"
            plt.figure(figsize=(8, 8))
            plt.imshow(heat, cmap="jet")
            plt.colorbar()
            plt.title(f"{slide_id}  prob={slide_prob:.3f}")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(heat_png, dpi=150)
            plt.close()

        for rank, i in enumerate(idx, start=1):
            rows.append(
                dict(
                    slide_id=slide_id,
                    rank=rank,
                    tile_path=tile_records[i]["tile_path"],
                    alpha=float(alpha[i]),
                    x=int(tile_records[i]["x"]),
                    y=int(tile_records[i]["y"]),
                    slide_prob=float(slide_prob),
                    y_true=int(r["label"]),
                )
            )

    out_csv = out_dir / "topk_tiles.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("Saved:", out_csv)

if __name__ == "__main__":
    main()
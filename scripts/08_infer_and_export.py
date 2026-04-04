import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm import tqdm
from torch.amp import autocast

from wsi_mil.datasets.bag_dataset import SlideBagDataset, build_transforms
from wsi_mil.models.wsi_mil_model import WSIBaselineMIL
from wsi_mil.utils.metrics import compute_metrics


def load_tile_batch(tile_records, transform, device):
    """加载一批图块到 GPU"""
    imgs = []
    for rec in tile_records:
        img = Image.open(rec["tile_path"]).convert("RGB")
        img = transform(img)
        imgs.append(img)
    batch = torch.stack(imgs, dim=0).to(device, non_blocking=True)
    return batch


@torch.no_grad()
def infer_slide_lazy(model, tile_records, device, transform, tile_bs: int = 64, amp: bool = True):
    """
    分块推理单个 slide，图块留在 CPU，每次只搬一小块到 GPU
    """
    model.eval()
    n_tiles = len(tile_records)
    
    # 分块编码：每次只加载 tile_bs 个图块到 GPU
    z_list = []
    for i in range(0, n_tiles, tile_bs):
        batch_records = tile_records[i:i+tile_bs]
        X = load_tile_batch(batch_records, transform, device)
        
        with autocast(device_type='cuda', enabled=amp):
            zb = model.encoder(X)
        z_list.append(zb.cpu())
        del X
        torch.cuda.empty_cache()
    
    # 合并特征，做 MIL
    z = torch.cat(z_list, dim=0).unsqueeze(0).to(device)
    
    with autocast(device_type='cuda', enabled=amp):
        slide_logit, alpha, h, _ = model.mil(z)
        slide_prob = torch.sigmoid(slide_logit).item()
    
    del z
    torch.cuda.empty_cache()
    
    return slide_prob, slide_logit.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/stage1_baseline.yaml")
    ap.add_argument("--ckpt", type=str, default="runs/stage1_baseline/ckpt_best.pt")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--tile_bs", type=int, default=64, help="GPU batch size for encoding")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    out_dir = Path(cfg["experiment"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    transform = build_transforms(train=False, img_size=cfg["data"]["img_size"])

    # 加载模型
    model = WSIBaselineMIL(**cfg["model"]).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)

    # lazy_load 模式：Dataset 只返回路径，不加载图像
    ds = SlideBagDataset(
        splits_csv=cfg["data"]["splits_csv"],
        bag_index_json=cfg["data"]["bag_index_json"],
        split=args.split,
        bag_size=cfg["data"]["bag_size"],
        img_size=cfg["data"]["img_size"],
        seed=cfg["experiment"]["seed"],
        full_bag=True,
        lazy_load=True,  # [关键] 延迟加载模式
    )
    
    # 直接迭代 Dataset，不使用 DataLoader（避免 collate_fn 转置问题）
    rows = []
    for i in tqdm(range(len(ds)), desc=f"infer {args.split}"):
        _, label, meta = ds[i]  # 直接访问，保持原始数据结构
        slide_id = meta["slide_id"]
        label_val = label.item()
        tile_records = meta["tile_records"]  # 正确格式：列表[Dict]
        
        if len(tile_records) == 0:
            continue

        slide_prob, _ = infer_slide_lazy(
            model, tile_records, device, transform,
            tile_bs=args.tile_bs, amp=cfg["train"]["amp"]
        )
        
        rows.append({"slide_id": slide_id, "y_true": label_val, "y_prob": slide_prob})

    pred_df = pd.DataFrame(rows)
    m = compute_metrics(pred_df["y_true"].tolist(), pred_df["y_prob"].tolist(), thr=0.5)
    metrics = {"loss": float("nan"), "auc": m.auc, "f1": m.f1, 
               "sensitivity": m.sensitivity, "specificity": m.specificity}

    pred_path = out_dir / f"preds_{args.split}.csv"
    pred_df.to_csv(pred_path, index=False)
    metric_path = out_dir / f"metrics_{args.split}.json"
    metric_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Saved:", pred_path, "auc=", metrics["auc"])


if __name__ == "__main__":
    main()
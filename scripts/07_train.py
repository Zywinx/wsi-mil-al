import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from wsi_mil.datasets.bag_dataset import SlideBagDataset
from wsi_mil.models.wsi_mil_model import WSIBaselineMIL
from wsi_mil.train.trainer import train_one_epoch, evaluate, save_ckpt
from wsi_mil.utils.seed import seed_everything
from wsi_mil.utils.log import get_logger
from wsi_mil.utils.io import mkdir

def collate_fn(batch):
    # batch_size=1 is simplest; keep generic
    bag_imgs, label, meta = zip(*batch)
    bag_imgs = torch.stack(bag_imgs, dim=0)
    label = torch.stack(label, dim=0)
    # meta to dict of lists
    meta_out = {}
    for k in meta[0].keys():
        meta_out[k] = [m[k] for m in meta]
    return bag_imgs, label, meta_out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/stage1_baseline.yaml")
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    out_dir = Path(cfg["experiment"]["out_dir"])
    mkdir(out_dir)
    logger = get_logger(str(out_dir / "train.log"))

    seed_everything(cfg["experiment"]["seed"], deterministic=False)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}")

    ds_train = SlideBagDataset(
        splits_csv=cfg["data"]["splits_csv"],
        bag_index_json=cfg["data"]["bag_index_json"],
        split="train",
        bag_size=cfg["data"]["bag_size"],
        img_size=cfg["data"]["img_size"],
        seed=cfg["experiment"]["seed"],
    )
    ds_val = SlideBagDataset(
        splits_csv=cfg["data"]["splits_csv"],
        bag_index_json=cfg["data"]["bag_index_json"],
        split="val",
        bag_size=cfg["data"]["bag_size"],  # eval also sample same size for speed
        img_size=cfg["data"]["img_size"],
        seed=cfg["experiment"]["seed"],
    )

    train_loader = DataLoader(
        ds_train,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = WSIBaselineMIL(**cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scaler = torch.amp.GradScaler('cuda', enabled=cfg["train"]["amp"])

    best_auc = -1.0
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device, scaler, amp=cfg["train"]["amp"])
        val = evaluate(model, val_loader, device, amp=cfg["train"]["amp"])

        logger.info(
            f"epoch={epoch} tr_loss={tr_loss:.4f} "
            f"val_loss={val['loss']:.4f} auc={val['auc']:.4f} f1={val['f1']:.4f} "
            f"sen={val['sensitivity']:.4f} spe={val['specificity']:.4f}"
        )

        save_ckpt(str(out_dir / "ckpt_last.pt"), model, optimizer, epoch, best_auc)

        if val["auc"] == val["auc"] and val["auc"] > best_auc:
            best_auc = val["auc"]
            save_ckpt(str(out_dir / "ckpt_best.pt"), model, optimizer, epoch, best_auc)
            (out_dir / "metrics_val.json").write_text(json.dumps(val, indent=2), encoding="utf-8")

    logger.info(f"best_auc={best_auc:.4f}")

if __name__ == "__main__":
    main()
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler 
from tqdm import tqdm

from wsi_mil.utils.metrics import compute_metrics
from wsi_mil.utils.io import mkdir

def save_ckpt(path: str, model, optimizer, epoch: int, best_metric: float):
    p = Path(path)
    mkdir(p.parent)
    torch.save(
        {
            "epoch": epoch,
            "best_metric": best_metric,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        p,
    )

@torch.no_grad()
def evaluate(model, loader, device, amp: bool = True, tile_bs: int = 128) -> Dict[str, Any]:
    """
    tile_bs: chunk size for encoding tiles when bag is large (to avoid OOM)
    """
    model.eval()
    y_true, y_prob, slide_ids = [], [], []
    losses = []

    for bag_imgs, label, meta in tqdm(loader, desc="eval", leave=False):
        # bag_imgs: [1, N, 3, H, W] where N can be very large when full_bag=True
        bag_imgs = bag_imgs.to(device, non_blocking=True)
        label = label.to(device)
        
        B, N = bag_imgs.shape[:2]
        
        with autocast('cuda',enabled=amp):
            # If bag is small, use standard forward
            if N <= tile_bs:
                out = model(bag_imgs)
            else:
                # Chunked encoding for large bags
                # Encode tiles in chunks to avoid OOM
                z_list = []
                for i in range(0, N, tile_bs):
                    chunk = bag_imgs[:, i:i+tile_bs].reshape(-1, *bag_imgs.shape[2:])  # [chunk_size, 3, H, W]
                    z_chunk = model.encoder(chunk)  # [chunk_size, D]
                    z_list.append(z_chunk)
                z = torch.cat(z_list, dim=0).unsqueeze(0)  # [1, N, D]
                # MIL aggregation
                slide_logit, alpha, h, _ = model.mil(z)
                slide_prob = torch.sigmoid(slide_logit)
                from wsi_mil.models.wsi_mil_model import ForwardOut
                out = ForwardOut(slide_logit=slide_logit, slide_prob=slide_prob, alpha=alpha, h=h, tile_z=z)
            
            logit = out.slide_logit
            loss = F.binary_cross_entropy_with_logits(logit, label.float())
        losses.append(loss.item())

        prob = out.slide_prob.detach().cpu().numpy().tolist()
        y_prob += prob
        y_true += label.detach().cpu().numpy().tolist()
        slide_ids += meta["slide_id"] if isinstance(meta["slide_id"], list) else [meta["slide_id"]]

    m = compute_metrics(y_true, y_prob, thr=0.5)
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "auc": m.auc,
        "f1": m.f1,
        "sensitivity": m.sensitivity,
        "specificity": m.specificity,
        "y_true": y_true,
        "y_prob": y_prob,
        "slide_id": slide_ids,
    }

def train_one_epoch(model, loader, optimizer, device, scaler: GradScaler, amp: bool = True) -> float:
    model.train()
    losses = []

    for bag_imgs, label, _meta in tqdm(loader, desc="train", leave=False):
        bag_imgs = bag_imgs.to(device, non_blocking=True)
        label = label.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=amp):
            out = model(bag_imgs)
            logit = out.slide_logit
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, label.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())

    return float(np.mean(losses)) if losses else float("nan")
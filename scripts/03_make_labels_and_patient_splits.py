import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides_csv", type=str, default="data/metadata/slides_raw.csv")
    ap.add_argument("--out_splits", type=str, default="data/metadata/splits_patient.csv")
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    df = pd.read_csv(args.slides_csv)
    # 正确逻辑
    required = {"slide_id", "label", "patient_id"}
    if not required.issubset(df.columns):
        df["slide_id"] = df["file_id"].astype(str)

    # We define slide_id = file_id (unique & stable) for engineering simplicity
    df["slide_id"] = df["file_id"].astype(str)

    # Patient-level split with GroupShuffleSplit
    gss1 = GroupShuffleSplit(n_splits=1, train_size=args.train_ratio, random_state=args.seed)
    train_idx, temp_idx = next(gss1.split(df, groups=df["patient_id"]))
    df_train = df.iloc[train_idx].copy()
    df_temp = df.iloc[temp_idx].copy()

    # Split temp into val/test
    # val_ratio is relative to full; compute relative within temp
    val_rel = args.val_ratio / (1.0 - args.train_ratio)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=val_rel, random_state=args.seed)
    val_idx, test_idx = next(gss2.split(df_temp, groups=df_temp["patient_id"]))
    df_val = df_temp.iloc[val_idx].copy()
    df_test = df_temp.iloc[test_idx].copy()

    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"

    out = pd.concat([df_train, df_val, df_test], axis=0).reset_index(drop=True)

    out_path = Path(args.out_splits)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print("Split sizes:", out["split"].value_counts().to_dict())
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
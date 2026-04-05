import json
import pandas as pd
from pathlib import Path
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile_manifest", type=str, default="data/metadata/tile_manifest.csv")
    ap.add_argument("--out_json", type=str, default="data/metadata/bag_index.json")
    ap.add_argument("--min_tiles", type=int, default=1, help="拦截切片数少于该值的 Slide") # 新增阈值参数
    args = ap.parse_args()

    df = pd.read_csv(args.tile_manifest)
    
    # 1. 全局排序
    df = df.sort_values(by=['slide_id', 'y', 'x', 'tile_path'])
    
    bags = {}
    slide_stats = []
    
    # 2. 高效分组
    for slide_id, group in df.groupby('slide_id'):
        
        cols_to_keep = ['tile_path', 'x', 'y', 'tile_size', 'tissue_ratio', 'source_wsi']
        tiles = group[cols_to_keep].to_dict(orient='records')
        n_tiles = len(tiles)
        
        # [补回并强化] 空包与极小包拦截逻辑
        if n_tiles < args.min_tiles:
            print(f" Warning: Slide {slide_id} 只有 {n_tiles} 个切片，低于阈值 {args.min_tiles}，已跳过。")
            continue
            
        bags[slide_id] = tiles
        
        slide_stats.append({
            "slide_id": slide_id,
            "n_tiles": n_tiles,
            "avg_tissue_ratio": group['tissue_ratio'].mean()
        })

    # 3. 保存与统计
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out, "w", encoding="utf-8") as f:
        json.dump(bags, f, ensure_ascii=False)
        
    stats_df = pd.DataFrame(slide_stats)
    stats_csv = out.parent / "bag_index_stats.csv"
    stats_df.to_csv(stats_csv, index=False)
    
    print(f"Saved index: {out} (n_slides={len(bags)})")
    print(f"Saved stats: {stats_csv}")
    print(f"Total tiles: {stats_df['n_tiles'].sum() if not stats_df.empty else 0}")

if __name__ == "__main__":
    main()
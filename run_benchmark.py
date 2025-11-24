from __future__ import annotations
import argparse
from runners.runner import run_all

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--max_samples", type=int, default=None)   
    args = ap.parse_args()

    res = run_all(dataset_path=args.dataset,
                  out_dir=args.out_dir,
                  max_samples=args.max_samples)                

    s = res["summary"]

    print("\n=== SUMMARY ===")
    print(f"provider/model : {s['provider']}/{s['model']}")
    print(f"samples        : {s['samples']}")
    print(f"base acc       : {s['base_acc']:.3f}")
    print(f"mani acc       : {s['mani_acc']:.3f}")
    print(f"follow_rate    : {s['follow_rate']:.3f}")
    print(f"core csv       : {s['core_csv']}")
  

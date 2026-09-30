"""
Verifies splits, checks class/box counts, calculates object scale metrics for target 
imgsz, and exports data/iid.yaml + data/ood.yaml. Fails fast (non-zero exit) on error.

python check_data.py
"""

import sys

import numpy as np

from common import (load_config, load_ground_truth, split_images_dir,
                    write_data_yamls)


def main() -> int:
    cfg = load_config()
    names = cfg["data"]["class_names"]
    ok = True

    for split in ["iid_train", "iid_val", "iid_test", "ood"]:
        d = split_images_dir(cfg, split)
        print(f"\n=== {split}: {d}")
        if not d.is_dir():
            print("  MISSING images directory")
            ok = False
            continue
        gts = load_ground_truth(d)
        if not gts:
            print("  no images found")
            ok = False
            continue
        cls = np.concatenate([g["cls"] for g in gts])
        boxes = np.concatenate([g["xyxy"] for g in gts])
        n_fg = sum(len(g["cls"]) > 0 for g in gts)
        print(f"  images {len(gts)} | with targets {n_fg} | background {len(gts) - n_fg}"
              f" | boxes {len(cls)}")
        bad = set(np.unique(cls)) - set(range(len(names)))
        if bad:
            print(f"  ERROR: class ids {sorted(bad)} are outside 0..{len(names) - 1}")
            ok = False
        for k, n in enumerate(names):
            print(f"    class {k} ({n}): {(cls == k).sum()} boxes")
        if len(boxes):
            side = np.sqrt((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
            long_side = np.array([max(g["w"], g["h"]) for g in gts for _ in g["cls"]])
            q = np.percentile(side, [10, 50, 90])
            print(f"  box size sqrt(area), native px: p10 {q[0]:.0f} | median {q[1]:.0f} | p90 {q[2]:.0f}")
            for sz in cfg["imgsz"]:
                s = side * sz / long_side
                q = np.percentile(s, [10, 50, 90])
                print(f"    after resize to {sz}: p10 {q[0]:.1f} | median {q[1]:.1f} | p90 {q[2]:.1f} px")
            a, b = cfg["eval"].get("size_bins_px", [64, 128])
            print(f"  size buckets sqrt(area) <{a} / {a}-{b} / >={b} px: small {(side < a).mean():.1%} | "
                  f"medium {((side >= a) & (side < b)).mean():.1%} | large {(side >= b).mean():.1%}")
        res = sorted({(g["w"], g["h"]) for g in gts})
        print(f"  image resolutions: {res[:5]}{' ...' if len(res) > 5 else ''}")
        if split == "ood":
            seqs = {}
            for g in gts:
                seqs.setdefault(g["seq"], [0, 0])
                seqs[g["seq"]][0] += 1
                seqs[g["seq"]][1] += len(g["cls"])
            print("  OOD sequences (images, boxes):")
            for s in sorted(seqs, key=lambda x: (len(x), x)):
                print(f"    {s}: {seqs[s][0]} images, {seqs[s][1]} boxes")
            if "unknown" in seqs:
                print("  WARNING: some OOD images have no US<N> token in their path; "
                      "per-sequence results will group them as 'unknown'.")

    iid_yaml, ood_yaml = write_data_yamls(cfg)
    print(f"\nWrote {iid_yaml}\nWrote {ood_yaml}")
    print("\nCheck by hand: class 0 must be PFM-1 and class 1 PMA-2 in EVERY split "
          "(SULAND v1 OOD labels had them reversed).")
    print("RESULT:", "OK" if ok else "PROBLEMS FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

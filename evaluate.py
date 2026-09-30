"""Evaluate trained models across all test splits.

Discovers finished runs (marked by DONE) and generates results/<run>/metrics.json.
Calculates standard Ultralytics metrics, COCO-style AP/AR across object scales, 
operating point performance (P/R/F1/FP at calibrated thresholds), per-sequence 
OOD breakdown (US1–US10), and model resource footprint (GFLOPs, parameters, latency).

python evaluate.py                 # Evaluate all completed runs
python evaluate.py --only v10s     # Filter by run name
python evaluate.py --force         # Recompute existing metrics
"""

import argparse
import json
import pickle
import time

import numpy as np

from common import (HERE, build_runs, load_config, load_ground_truth, run_dir,
                    split_images_dir, write_data_yamls)
from metrics_lib import (at_threshold, choose_thresholds, coco_eval,
                         match_flags, preds_from_val_json)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--device", default=None)
    return p.parse_args()


def ultra_val(model, data_yaml, split, imgsz, e, device, save_dir, names):
    """Official ultralytics validation. Also writes predictions.json, which is
    the single source of detections for every other metric in this study."""
    name = f"val_{split}_{data_yaml.stem}"
    json_path = save_dir / name / "predictions.json"
    json_path.unlink(missing_ok=True)
    m = model.val(data=str(data_yaml), split=split, imgsz=imgsz, batch=e["batch"],
                  conf=e["conf"], iou=e["nms_iou"], max_det=e["max_det"],
                  device=device, plots=False, verbose=False, save_json=True,
                  project=str(save_dir), name=name, exist_ok=True)
    b = m.box
    out = {"mAP50": float(b.map50), "mAP50_95": float(b.map), "mAP75": float(b.map75),
           "precision": float(b.mp), "recall": float(b.mr)}
    for i, c in enumerate(b.ap_class_index):
        out[f"AP50_{names[int(c)]}"] = float(b.ap50[i])
        out[f"AP50_95_{names[int(c)]}"] = float(b.ap[i])
        out[f"P_{names[int(c)]}"] = float(b.p[i])
        out[f"R_{names[int(c)]}"] = float(b.r[i])
    out["speed_ms"] = {k: float(v) for k, v in m.speed.items()}
    if not json_path.exists():
        raise FileNotFoundError(f"validator did not write {json_path}")
    return out, json_path


def cost(model, imgsz, gts, e, device, save_dir):
    from ultralytics.utils.torch_utils import get_flops
    net = model.model
    out = {"params_M": sum(p.numel() for p in net.parameters()) / 1e6}
    try:
        out["GFLOPs"] = float(get_flops(net, imgsz))
    except Exception as ex:          # FLOP counting needs `thop`; not critical
        out["GFLOPs"] = float("nan")
        out["GFLOPs_error"] = str(ex)
    paths = [g["path"] for g in gts][: e["latency_images"]]
    tmp = dict(save=False, project=str(save_dir), name="latency", exist_ok=True)
    for p in paths[:20]:                                   # warm-up
        model.predict(p, imgsz=imgsz, device=device, verbose=False, **tmp)
    pre, inf, post = [], [], []
    t0 = time.perf_counter()
    for p in paths:
        r = model.predict(p, imgsz=imgsz, conf=e["fixed_conf"], device=device,
                          verbose=False, **tmp)[0]
        pre.append(r.speed["preprocess"]); inf.append(r.speed["inference"])
        post.append(r.speed["postprocess"])
    wall = (time.perf_counter() - t0) / max(len(paths), 1) * 1000
    out.update({"latency_ms_preprocess": float(np.mean(pre)),
                "latency_ms_inference": float(np.mean(inf)),
                "latency_ms_postprocess": float(np.mean(post)),
                "latency_ms_total": float(np.mean(pre) + np.mean(inf) + np.mean(post)),
                "latency_ms_wall": wall, "fps_batch1": 1000 / max(wall, 1e-9)})
    return out


def evaluate_run(cfg, run, device, force):
    e, names = cfg["eval"], cfg["data"]["class_names"]
    rdir = run_dir(cfg, run)
    weights = rdir / "weights" / "best.pt"
    out_dir = HERE / e["results_dir"] / run.name
    out_file = out_dir / "metrics.json"
    if not weights.exists():
        print(f"[skip] {run.name}: no best.pt yet")
        return
    if out_file.exists() and not force:
        print(f"[skip] {run.name}: already evaluated (use --force)")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO
    model = YOLO(str(weights))
    iid_yaml, ood_yaml = write_data_yamls(cfg)
    res = {"run": run.name, "model": run.model_tag, "imgsz": run.imgsz, "aug": run.aug,
           "seed": run.seed, "weights": str(weights)}

    print(f"\n===== {run.name} =====")
    print("  ultralytics val ...")
    res["ultralytics"], json_paths = {}, {}
    for key, yml, split in [("iid_val", iid_yaml, "val"), ("iid_test", iid_yaml, "test"),
                            ("ood", ood_yaml, "val")]:
        res["ultralytics"][key], json_paths[key] = ultra_val(
            model, yml, split, run.imgsz, e, device, out_dir, names)

    preds, gts = {}, {}
    for split in ["iid_val", "iid_test", "ood"]:
        gts[split] = load_ground_truth(split_images_dir(cfg, split))
        preds[split] = preds_from_val_json(json_paths[split], gts[split])
    with open(out_dir / "predictions.pkl", "wb") as f:       # for later re-analysis
        pickle.dump({"preds": preds, "gts": gts}, f)

    bins = tuple(e.get("size_bins_px", (64, 128)))
    res["size_bins_px"] = list(bins)
    flags = {s: match_flags(preds[s], gts[s], e["match_iou"]) for s in preds}
    thr = choose_thresholds(flags["iid_val"], e["precision_target"])
    thr["fixed"] = e["fixed_conf"]
    res["thresholds_from_iid_val"] = thr

    res["coco"], res["operating"] = {}, {}
    for split in ["iid_test", "ood"]:
        print(f"  COCO + operating points on {split} ...")
        res["coco"][split] = coco_eval(preds[split], gts[split], names, size_bins=bins)
        res["operating"][split] = {k: at_threshold(flags[split], len(gts[split]), v, len(names))
                                   for k, v in thr.items()}

    print("  per-sequence OOD ...")
    seqs = {}
    for i, g in enumerate(gts["ood"]):
        seqs.setdefault(g["seq"], []).append(i)
    res["per_sequence"] = {}
    for s, idx in sorted(seqs.items(), key=lambda kv: (len(kv[0]), kv[0])):
        c = coco_eval(preds["ood"], gts["ood"], names, subset=idx, size_bins=bins)
        sub_p = [preds["ood"][i] for i in idx]
        sub_g = [gts["ood"][i] for i in idx]
        op = at_threshold(match_flags(sub_p, sub_g, e["match_iou"]), len(idx),
                          thr["f1_max"], len(names))
        res["per_sequence"][s] = {"images": len(idx),
                                  "boxes": int(sum(len(gts["ood"][i]["cls"]) for i in idx)),
                                  "AP50": c["AP50"], "AP": c["AP"],
                                  "recall_at_f1thr": op["recall"],
                                  "FP_per_image_at_f1thr": op["FP_per_image"]}

    print("  cost ...")
    res["cost"] = cost(model, run.imgsz, gts["ood"], e, device, out_dir)
    out_file.write_text(json.dumps(res, indent=2, default=float))
    u = res["ultralytics"]
    print(f"  IID test mAP50 {u['iid_test']['mAP50']:.3f} | OOD mAP50 {u['ood']['mAP50']:.3f}"
          f" | OOD recall {u['ood']['recall']:.3f} -> {out_file}")


def main():
    a = parse_args()
    cfg = load_config()
    runs = build_runs(cfg)
    if a.only:
        runs = [r for r in runs if any(s in r.name for s in a.only)]
    device = a.device or cfg["train"]["device"].split(",")[0]
    for r in runs:
        if not (run_dir(cfg, r) / "DONE").exists():
            print(f"[skip] {r.name}: training not finished")
            continue
        evaluate_run(cfg, r, device, a.force)


if __name__ == "__main__":
    main()

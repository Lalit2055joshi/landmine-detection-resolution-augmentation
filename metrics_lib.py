"""Metrics that the SULAND_v2 benchmark did not report.

- COCO-style AP by object size (small / medium / large), AP75, AR
- operating-point metrics: precision, recall, F1, false positives per image,
  missed targets, at thresholds chosen on IID val (never on the test split)
- per-sequence OOD AP50
"""

from __future__ import annotations

import contextlib
import io

import numpy as np


# ----------------------------------------------------------------- predictions
def preds_from_val_json(json_path, gts: list[dict]) -> list[dict]:
    """Load the detections written by ultralytics' validator (save_json=True),
    aligned with gts order.

    Using the validator's own detections guarantees that every extra metric is
    computed from exactly the predictions behind the headline mAP. model.predict()
    preprocesses differently (resize interpolation, letterbox padding) and was
    measured to give OOD mAP50 ~8 points lower than model.val() on SULAND_v2.
    Boxes in the json are COCO xywh (top-left) in native image pixels and
    category_id = class index + 1.
    """
    import json
    from pathlib import Path

    names = [Path(g["path"]).name for g in gts]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate image file names in split; cannot align {json_path}")
    index = {n: i for i, n in enumerate(names)}
    per_img = [[] for _ in gts]
    unknown = 0
    for d in json.loads(Path(json_path).read_text()):
        name = d.get("file_name") or f"{d['image_id']}"
        i = index.get(name)
        if i is None:                       # older versions: match on stem
            i = next((j for j, n in enumerate(names) if Path(n).stem == str(d["image_id"])), None)
        if i is None:
            unknown += 1
            continue
        x, y, w, h = d["bbox"]
        per_img[i].append((x, y, x + w, y + h, d["score"], d["category_id"] - 1))
    if unknown:
        raise ValueError(f"{unknown} detections in {json_path} match no image of the split")
    out = []
    for rows in per_img:
        a = np.asarray(rows, np.float32).reshape(-1, 6)
        out.append({"xyxy": a[:, :4], "conf": a[:, 4], "cls": a[:, 5].astype(int)})
    return out


# ----------------------------------------------------------------- matching
def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(2)
    area_a = (a[:, 2:] - a[:, :2]).prod(1)
    area_b = (b[:, 2:] - b[:, :2]).prod(1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match_flags(preds: list[dict], gts: list[dict], iou_thr: float = 0.5):
    """Greedy, class-aware, confidence-ordered matching per image.

    Because higher-confidence predictions are matched first, the TP/FP flag of
    each prediction does not depend on lower-confidence ones, so the flags can
    be computed once and reused for any confidence threshold.

    Returns flat arrays conf, cls, tp, img_idx, and the GT class array.
    """
    conf_all, cls_all, tp_all, img_all = [], [], [], []
    for i, (p, g) in enumerate(zip(preds, gts)):
        order = np.argsort(-p["conf"])
        pc, pk, pb = p["conf"][order], p["cls"][order], p["xyxy"][order]
        tp = np.zeros(len(pc), bool)
        if len(pc) and len(g["cls"]):
            iou = _iou(pb, g["xyxy"])
            used = np.zeros(len(g["cls"]), bool)
            for j in range(len(pc)):
                cand = np.where((g["cls"] == pk[j]) & ~used & (iou[j] >= iou_thr))[0]
                if len(cand):
                    best = cand[np.argmax(iou[j, cand])]
                    used[best] = True
                    tp[j] = True
        conf_all.append(pc); cls_all.append(pk); tp_all.append(tp)
        img_all.append(np.full(len(pc), i))
    gt_cls = np.concatenate([g["cls"] for g in gts]) if gts else np.zeros(0, int)
    cat = lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dt)
    return (cat(conf_all, np.float32), cat(cls_all, int), cat(tp_all, bool),
            cat(img_all, int), gt_cls)


def at_threshold(flags, n_images: int, thr: float, n_classes: int) -> dict:
    conf, cls, tp, _, gt_cls = flags
    keep = conf >= thr
    TP, FP = int(tp[keep].sum()), int((~tp[keep]).sum())
    n_gt = len(gt_cls)
    P = TP / max(TP + FP, 1)
    R = TP / max(n_gt, 1)
    out = {"threshold": float(thr), "precision": P, "recall": R,
           "F1": 2 * P * R / max(P + R, 1e-9), "TP": TP, "FP": FP,
           "FN": n_gt - TP, "FP_per_image": FP / max(n_images, 1)}
    for k in range(n_classes):
        m = keep & (cls == k)
        tpk = int(tp[m].sum())
        out[f"recall_c{k}"] = tpk / max(int((gt_cls == k).sum()), 1)
        out[f"precision_c{k}"] = tpk / max(int(m.sum()), 1)
    return out


def choose_thresholds(flags, precision_target: float) -> dict:
    """From IID-val flags: the F1-maximising threshold, and the lowest threshold
    whose precision is >= precision_target (i.e. the highest recall there)."""
    conf, _, tp, _, gt_cls = flags
    order = np.argsort(-conf, kind="stable")
    c, t = conf[order], tp[order]
    ctp = np.cumsum(t)
    cfp = np.cumsum(~t)
    # A threshold keeps every prediction with conf >= thr, so ties must enter
    # together: evaluate only at the last index of each group of equal conf.
    last = np.r_[c[1:] != c[:-1], True] if len(c) else np.zeros(0, bool)
    c, ctp, cfp = c[last], ctp[last], cfp[last]
    prec = ctp / np.maximum(ctp + cfp, 1)
    rec = ctp / max(len(gt_cls), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    out = {"f1_max": float(c[np.argmax(f1)]) if len(c) else 0.25}
    ok = np.where(prec >= precision_target)[0]
    out["precision_target"] = float(c[ok.max()]) if len(ok) else float(c[0]) if len(c) else 0.25
    return out


# ----------------------------------------------------------------- COCO eval
def coco_eval(preds: list[dict], gts: list[dict], class_names: list[str],
              subset: list[int] | None = None, size_bins=(32, 96)) -> dict:
    """COCO metrics (AP, AP50, AP75, AP by size, AR) with pycocotools.

    size_bins = (a, b): an object is small if sqrt(area) < a, medium if
    a <= sqrt(area) < b, large otherwise, in NATIVE image pixels. The COCO
    defaults (32, 96) put almost every SULAND mine in "large", so config.yaml
    sets bins suited to this data.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    idx = subset if subset is not None else range(len(gts))
    images, anns, dets = [], [], []
    aid = 1
    for i in idx:
        g, p = gts[i], preds[i]
        images.append({"id": i, "width": g["w"], "height": g["h"]})
        for k, b in zip(g["cls"], g["xyxy"]):
            w, h = float(b[2] - b[0]), float(b[3] - b[1])
            anns.append({"id": aid, "image_id": i, "category_id": int(k) + 1,
                         "bbox": [float(b[0]), float(b[1]), w, h], "area": w * h,
                         "iscrowd": 0})
            aid += 1
        for k, s, b in zip(p["cls"], p["conf"], p["xyxy"]):
            dets.append({"image_id": i, "category_id": int(k) + 1, "score": float(s),
                         "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]})
    keys = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
            "AR1", "AR10", "AR300", "AR_small", "AR_medium", "AR_large"]
    if not anns or not dets:
        return {k: float("nan") for k in keys}

    gt = COCO()
    gt.dataset = {"images": images, "annotations": anns,
                  "categories": [{"id": k + 1, "name": n} for k, n in enumerate(class_names)]}
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
        dt = gt.loadRes(dets)
        ev = COCOeval(gt, dt, "bbox")
        ev.params.maxDets = [1, 10, 300]      # keep all detections, like ultralytics
        a, b = size_bins
        ev.params.areaRng = [[0, 1e10], [0, a * a], [a * a, b * b], [b * b, 1e10]]
        ev.evaluate(); ev.accumulate(); ev.summarize()
    out = {k: (float(v) if v >= 0 else float("nan")) for k, v in zip(keys, ev.stats)}
    # precision array is [T, R, K, A, M]. summarize() computes stats[0] with
    # maxDets=100, which is not in our list, so recompute AP at 300 detections.
    prec = ev.eval["precision"]
    allp = prec[:, :, :, 0, -1]
    out["AP"] = float(allp[allp > -1].mean()) if (allp > -1).any() else float("nan")
    for k, n in enumerate(class_names):
        pk = prec[0, :, k, 0, -1]
        out[f"AP50_{n}"] = float(pk[pk > -1].mean()) if (pk > -1).any() else float("nan")
    return out

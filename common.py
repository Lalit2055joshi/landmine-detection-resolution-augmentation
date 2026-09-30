"""Shared helpers: config loading, dataset yamls, run grid, label parsing."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

HERE = Path(__file__).resolve().parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SEQ_RE = re.compile(r"US[_\-]?0*(\d+)", re.IGNORECASE)


def load_config(path: str | Path | None = None) -> dict:
    """config.yaml next to this file, or the file named by $STUDY_CONFIG."""
    path = path or os.environ.get("STUDY_CONFIG") or HERE / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


# ----------------------------------------------------------------- data yamls
def write_data_yamls(cfg: dict) -> tuple[Path, Path]:
    """Write data/iid.yaml and data/ood.yaml with ABSOLUTE paths.

    ultralytics resolves a relative `path:` against its own datasets_dir, not
    against the yaml's location, so relative yamls silently point elsewhere.
    """
    d = cfg["data"]
    names = {i: n for i, n in enumerate(d["class_names"])}
    out = HERE / "data"
    out.mkdir(exist_ok=True)

    iid = Path(d["iid_root"]).resolve()
    s = d["iid_splits"]
    iid_yaml = out / "iid.yaml"
    iid_yaml.write_text(yaml.safe_dump({
        "path": str(iid),
        "train": f"{s['train']}/images",
        "val": f"{s['val']}/images",
        "test": f"{s['test']}/images",
        "names": names,
    }, sort_keys=False))

    ood = Path(d["ood_root"]).resolve()
    o = d["ood_split"]
    ood_yaml = out / "ood.yaml"
    ood_yaml.write_text(yaml.safe_dump({
        "path": str(ood),
        "train": f"{o}/images",       # required key; never trained on
        "val": f"{o}/images",
        "names": names,
    }, sort_keys=False))
    return iid_yaml, ood_yaml


def split_images_dir(cfg: dict, split: str) -> Path:
    """split in {iid_train, iid_val, iid_test, ood}."""
    d = cfg["data"]
    if split == "ood":
        return Path(d["ood_root"]) / d["ood_split"] / "images"
    key = split.replace("iid_", "")
    return Path(d["iid_root"]) / d["iid_splits"][key] / "images"


# ----------------------------------------------------------------- run grid
@dataclass(frozen=True)
class Run:
    model_tag: str
    weights: str
    imgsz: int
    aug: str
    seed: int

    @property
    def name(self) -> str:
        return f"{self.model_tag}_{self.imgsz}_{self.aug}_s{self.seed}"


def build_runs(cfg: dict) -> list[Run]:
    runs = []
    for seed in cfg["seeds"]:
        for tag, w in cfg["models"].items():
            for sz in cfg["imgsz"]:
                for aug in cfg["augmentation"]:
                    runs.append(Run(tag, w, int(sz), aug, int(seed)))
    return runs


def run_dir(cfg: dict, run: Run) -> Path:
    return HERE / cfg["train"]["project"] / run.name


# ----------------------------------------------------------------- images/labels
def list_images(images_dir: Path) -> list[Path]:
    return sorted(p for p in Path(images_dir).rglob("*") if p.suffix.lower() in IMG_EXTS)


def label_path(img: Path) -> Path:
    """Swap the last 'images' path component for 'labels' (ultralytics rule)."""
    parts = list(img.parts)
    idx = max(i for i, p in enumerate(parts) if p == "images")
    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")


def sequence_of(img: Path) -> str:
    """SULAND OOD sequence id (US1..US10) from folder or file name."""
    for part in reversed(img.parts):
        m = SEQ_RE.search(part)
        if m:
            return f"US{int(m.group(1))}"
    return "unknown"


def read_yolo_labels(txt: Path, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Return cls (N,) int and xyxy (N,4) float in original-image pixels."""
    if not txt.exists():
        return np.zeros(0, int), np.zeros((0, 4), np.float32)
    rows = []
    for line in txt.read_text().splitlines():
        v = line.split()
        if len(v) >= 5:
            rows.append([float(x) for x in v[:5]])
    if not rows:
        return np.zeros(0, int), np.zeros((0, 4), np.float32)
    a = np.asarray(rows, np.float32)
    cx, cy, bw, bh = a[:, 1] * w, a[:, 2] * h, a[:, 3] * w, a[:, 4] * h
    xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
    return a[:, 0].astype(int), xyxy


def load_ground_truth(images_dir: Path) -> list[dict]:
    gts = []
    for p in list_images(images_dir):
        with Image.open(p) as im:
            w, h = im.size
        cls, xyxy = read_yolo_labels(label_path(p), w, h)
        gts.append({"path": str(p), "w": w, "h": h, "cls": cls, "xyxy": xyxy,
                    "seq": sequence_of(p)})
    return gts

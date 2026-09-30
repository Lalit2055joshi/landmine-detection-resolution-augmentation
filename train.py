"""Train every run in the grid (or a subset).

Finished runs are skipped; interrupted runs resume from last.pt. Checkpoint
selection (best.pt) uses IID val only.

python train.py --list                       # show the runs
python train.py                              # train all, one after another
python train.py --only v8s_1024_aug_s0       # one run
python train.py --only v8s_640 --device 1    # every run whose name contains 'v8s_640'
"""

import argparse
import json
import platform
from pathlib import Path

from common import build_runs, load_config, run_dir, write_data_yamls


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", default=None,
                   help="substring(s) of run names to train")
    p.add_argument("--device", default=None, help="override train.device")
    p.add_argument("--list", action="store_true")
    p.add_argument("--epochs", type=int, default=None,
                   help="override epochs (e.g. 1 for a smoke test)")
    return p.parse_args()


def main():
    a = parse_args()
    cfg = load_config()
    t = cfg["train"]
    runs = build_runs(cfg)
    if a.only:
        runs = [r for r in runs if any(s in r.name for s in a.only)]
    if a.list or not runs:
        for r in runs:
            state = "done" if (run_dir(cfg, r) / "DONE").exists() else "todo"
            print(f"{r.name:28s} {state}")
        if not runs:
            print("no runs match")
        return

    from ultralytics import YOLO
    import ultralytics
    import torch

    iid_yaml, _ = write_data_yamls(cfg)
    epochs = a.epochs or t["epochs"]

    for r in runs:
        out = run_dir(cfg, r)
        if (out / "DONE").exists():
            print(f"[skip] {r.name} already finished")
            continue
        last = out / "weights" / "last.pt"
        print(f"\n===== {r.name} =====")
        if last.exists():
            print(f"[resume] from {last}")
            YOLO(str(last)).train(resume=True)
        else:
            args = dict(
                data=str(iid_yaml), imgsz=r.imgsz, epochs=epochs,
                batch=t["batch"][r.imgsz], patience=t["patience"],
                optimizer=t["optimizer"], lr0=t["lr0"], momentum=t["momentum"],
                weight_decay=t["weight_decay"], warmup_epochs=t["warmup_epochs"],
                device=a.device or t["device"], workers=t["workers"],
                cache=t["cache"], amp=t["amp"], seed=r.seed,
                deterministic=t["deterministic"], project=str(out.parent),
                name=r.name, exist_ok=True, plots=True, val=True,
                **cfg["augmentation"][r.aug],
            )
            YOLO(r.weights).train(**args)

        (out / "run_info.json").write_text(json.dumps({
            "run": r.name, "model": r.weights, "imgsz": r.imgsz, "aug": r.aug,
            "seed": r.seed, "epochs": epochs, "augmentation": cfg["augmentation"][r.aug],
            "train_cfg": t, "ultralytics": ultralytics.__version__,
            "torch": torch.__version__, "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        }, indent=2))
        (out / "DONE").touch()
        print(f"[done] {r.name} -> {out / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()

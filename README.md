# Resolution & Augmentation Study on SULAND_v2

Quick context: the SULAND_v2 paper (Lekhak et al., arXiv 2607.28996) trained
everything at 640px with no augmentation, and only reported mAP50, mAP50-95,
precision, recall, params and FPS. We wanted to know what happens if you
actually vary resolution and augmentation instead of holding them fixed, and
also pull out some metrics they didn't report (per-size AP, false positives
per image, that kind of thing).

So: 2 models × 2 resolutions × 2 augmentation settings = 8 runs.

| Factor | Levels |
|---|---|
| Model | YOLOv8-S (the SULAND reference model), YOLOv10-S (never evaluated on SULAND before) |
| Input size | 640, 1024 |
| Augmentation | `noaug` (matches the benchmark), `aug` (standard ultralytics recipe) |

Every run trains on IID train, picks its checkpoint off IID val only, and gets
scored on both IID test and OOD. Nothing from OOD touches training or model
selection.

Extra metrics we added on top of the benchmark: AP75, AP broken down by object
size (small/medium/large), AR, false positives per image, missed targets,
recall at 90% precision, per-sequence OOD AP50 (US1 through US10), GFLOPs and
latency at each resolution, and per-class numbers.

---

## What's in here

- `config.yaml` — the only file you should need to touch. Data paths, the grid, hyperparams.
- `check_data.py` — step 1. Sanity-checks the dataset, writes out `data/iid.yaml` and `data/ood.yaml`.
- `train.py` — step 2. Trains the whole grid. Skips runs that already finished, resumes ones that got interrupted.
- `evaluate.py` — step 3. Dumps every metric per run into `results/<run>/metrics.json`.
- `collect.py` — step 4. Builds the tables (`results/tables.md` + CSVs) and figures.
- `metrics_lib.py`, `common.py` — shared helper code, nothing to run directly.
- `run_all.sh` / `run_2gpu.sh` — runs the whole pipeline, either on one GPU or splitting across two.

---

## Running it on the server

### 0. Get it up there and install

```bash
python -m venv .venv && source .venv/bin/activate      # conda's fine too
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # match your CUDA version!
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 1. Get the data in place and point the config at it

Layout should look like this (SULAND_v2, YOLO-format txt labels, class 0 =
PFM-1 and class 1 = PMA-2 everywhere — don't let this flip between splits):

```
SULAND_v2/
  iid/train/images/...   iid/train/labels/...
  iid/val/images/...     iid/val/labels/...
  iid/test/images/...    iid/test/labels/...
  ood/val/images/US1/... ood/val/labels/US1/...   (US1..US10 — subfolders, or the US<N> token in filenames, either works)
```

Then set `data.iid_root` and `data.ood_root` in `config.yaml` to absolute paths.

### 2. Check the data first (no GPU needed, takes under a minute)

```bash
python check_data.py
```

Should end with `RESULT: OK`. Worth actually comparing the printed counts
against the SULAND_v2 numbers — IID train 22,756 images / 6,714 boxes, IID val
2,836 / 807, IID test 3,743 / 1,056, OOD 4,436 / 3,856 — and check all 10 OOD
sequences show up. It also prints box sizes after resizing to 640 and 1024,
which is worth keeping for the paper.


### 3. The actual training run

Use tmux, screen, or nohup — you don't want this dying when you log out.

```bash
nohup bash run_all.sh  > logs/run_all.log  2>&1 &     # single GPU
# or, if you've got two:
nohup bash run_2gpu.sh > logs/run_2gpu.log 2>&1 &      # v8s on GPU 0, v10s on GPU 1
tail -f logs/run_all.log
python train.py --list      # shows what's done and what's still pending
```

If it dies for whatever reason, just run the same command again — it'll pick
up where it left off instead of starting over.

A note on batch size: stick to values that divide 64 (8, 16, 32, 64).
Ultralytics accumulates `round(64/batch)` steps before updating, so all of
those give you an effective batch of 64 and stay comparable — 48 gives you 48,
24 gives you 72, so avoid those unless you mean it. On a 24GB card, 16 at 1024
worked for us (32 pushed close to 23GB in testing). If you OOM at 1024, drop
to 8.

Rough timing at 50 epochs with early stopping on something like an A100,
4090, or 3090: 2-4h for a 640 run, 5-9h for a 1024 run. So call it 1.5-2.5
days for all 8 if you're running them sequentially. `cache: ram` speeds
things up noticeably if the box has more than 64GB of RAM to spare.




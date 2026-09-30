"""
Compiles results/*/metrics.json into all_runs.csv, formatted markdown tables 
(tables.md), effect size metrics (effects.csv), and evaluation plots (OOD mAP50, 
AP by object size, per-sequence breakdown).

python collect.py
"""

import json

import numpy as np
import pandas as pd

from common import HERE, load_config

KEY = ["model", "imgsz", "aug"]


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        name = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, name + "."))
        elif isinstance(v, (int, float, str)) or v is None:
            out[name] = v
    return out


def fmt(mean, std, pct=True):
    s = 100 if pct else 1
    if np.isnan(mean):
        return "–"
    if np.isnan(std) or std == 0:
        return f"{mean * s:.1f}" if pct else f"{mean:.2f}"
    return f"{mean * s:.1f} ± {std * s:.1f}" if pct else f"{mean:.2f} ± {std:.2f}"


def table(df, cols, titles, pct=True):
    g = df.groupby(KEY)
    m, s = g[cols].mean(), g[cols].std()
    head = "| Model | imgsz | Aug | " + " | ".join(titles) + " |"
    sep = "|" + "---|" * (3 + len(cols))
    rows = [head, sep]
    for idx in m.index:
        cells = [fmt(m.loc[idx, c], s.loc[idx, c], pct) for c in cols]
        rows.append(f"| {idx[0]} | {idx[1]} | {idx[2]} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main():
    cfg = load_config()
    rdir = HERE / cfg["eval"]["results_dir"]
    names = cfg["data"]["class_names"]
    recs = [json.loads(p.read_text()) for p in sorted(rdir.glob("*/metrics.json"))]
    if not recs:
        print("no results yet — run evaluate.py first")
        return
    df = pd.DataFrame([flatten(r) for r in recs])
    df.to_csv(rdir / "all_runs.csv", index=False)
    n_seeds = df.groupby(KEY).size().max()

    U = "ultralytics."
    md = [f"# Results ({len(df)} runs, up to {n_seeds} seed(s) per configuration)\n",
          "Values are percentages unless stated. mean ± std across seeds when available.\n"]

    md += ["## Table 1 — Detection accuracy (ultralytics val, conf 0.001)\n",
           table(df, [U + "iid_test.mAP50", U + "iid_test.mAP50_95", U + "ood.mAP50",
                      U + "ood.mAP50_95", U + "ood.mAP75", U + "ood.precision", U + "ood.recall"],
                 ["IID mAP50", "IID mAP50-95", "OOD mAP50", "OOD mAP50-95", "OOD mAP75",
                  "OOD P", "OOD R"]), ""]

    md += ["## Table 2 — OOD per-class AP50\n",
           table(df, [f"{U}ood.AP50_{n}" for n in names] + [f"{U}iid_test.AP50_{n}" for n in names],
                 [f"OOD {n}" for n in names] + [f"IID {n}" for n in names]), ""]

    bins = cfg["eval"].get("size_bins_px", [64, 128])
    md += [f"## Table 3 — COCO AP 50:95 by object size (sqrt(area) < {bins[0]} / {bins[0]}–{bins[1]} / ≥ {bins[1]} native px)\n",
           table(df, ["coco.iid_test.AP_small", "coco.iid_test.AP_medium",
                      "coco.ood.AP_small", "coco.ood.AP_medium", "coco.ood.AP_large",
                      "coco.ood.AR300"],
                 ["IID AP-S", "IID AP-M", "OOD AP-S", "OOD AP-M", "OOD AP-L", "OOD AR@300"]), ""]

    for op, label in [("f1_max", "F1-max threshold (chosen on IID val)"),
                      ("precision_target", f"threshold giving {cfg['eval']['precision_target']:.0%} precision on IID val"),
                      ("fixed", f"fixed conf {cfg['eval']['fixed_conf']}")]:
        c = f"operating.ood.{op}."
        md += [f"## Table 4{'abc'[['f1_max', 'precision_target', 'fixed'].index(op)]} — OOD at {label}\n",
               table(df, [c + "precision", c + "recall", c + "F1"] +
                     [c + f"recall_c{k}" for k in range(len(names))],
                     ["P", "R", "F1"] + [f"R {n}" for n in names]),
               "",
               table(df, [c + "FP_per_image", c + "FN", c + "threshold"],
                     ["FP / image", "Missed targets", "Threshold"], pct=False), ""]

    md += ["## Table 5 — Cost\n",
           table(df, ["cost.params_M", "cost.GFLOPs", "cost.latency_ms_inference",
                      "cost.latency_ms_total", "cost.fps_batch1"],
                 ["Params (M)", "GFLOPs", "Inference ms", "Total ms", "FPS (bs 1)"], pct=False), ""]

    seq_cols = sorted({c for c in df.columns if c.startswith("per_sequence.") and c.endswith(".AP50")},
                      key=lambda c: (len(c), c))
    if seq_cols:
        md += ["## Table 6 — OOD AP50 per sequence\n",
               table(df, seq_cols, [c.split(".")[1] for c in seq_cols]), ""]

    # effects: aug - noaug and 1024 - 640, per model (paired on seed)
    eff_rows = []
    metrics = {"OOD mAP50": U + "ood.mAP50", "OOD mAP50-95": U + "ood.mAP50_95",
               "OOD recall": U + "ood.recall", "IID mAP50": U + "iid_test.mAP50",
               "OOD AP-small": "coco.ood.AP_small"}
    sizes = sorted(df["imgsz"].unique())
    augs = list(cfg["augmentation"])
    for model in df["model"].unique():
        d = df[df["model"] == model]
        if "aug" in augs and "noaug" in augs:
            for sz in sizes:
                a = d[(d.imgsz == sz) & (d.aug == "aug")].set_index("seed")
                b = d[(d.imgsz == sz) & (d.aug == "noaug")].set_index("seed")
                common = a.index.intersection(b.index)
                if len(common):
                    eff_rows.append({"model": model, "effect": f"augmentation @ {sz}",
                                     **{k: float((a.loc[common, v] - b.loc[common, v]).mean() * 100)
                                        for k, v in metrics.items()}})
        if len(sizes) >= 2:
            for ag in augs:
                a = d[(d.imgsz == sizes[-1]) & (d.aug == ag)].set_index("seed")
                b = d[(d.imgsz == sizes[0]) & (d.aug == ag)].set_index("seed")
                common = a.index.intersection(b.index)
                if len(common):
                    eff_rows.append({"model": model, "effect": f"{sizes[-1]} vs {sizes[0]} ({ag})",
                                     **{k: float((a.loc[common, v] - b.loc[common, v]).mean() * 100)
                                        for k, v in metrics.items()}})
    if eff_rows:
        eff = pd.DataFrame(eff_rows)
        eff.to_csv(rdir / "effects.csv", index=False)
        md += ["## Table 7 — Effects (percentage-point change, paired by seed)\n",
               "| Model | Effect | " + " | ".join(metrics) + " |",
               "|" + "---|" * (2 + len(metrics))]
        for _, r in eff.iterrows():
            md.append(f"| {r.model} | {r.effect} | " +
                      " | ".join(f"{r[k]:+.1f}" for k in metrics) + " |")
        md.append("")

    (rdir / "tables.md").write_text("\n".join(md))
    print((rdir / "tables.md").read_text())
    figures(df, rdir, seq_cols)


def figures(df, rdir, seq_cols):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = df.groupby(KEY).mean(numeric_only=True).reset_index()
    labels = [f"{r.model}\n{r.imgsz}\n{r.aug}" for r in g.itertuples()]
    x = np.arange(len(g))

    fig, ax = plt.subplots(figsize=(max(6, len(g) * 0.9), 3.6))
    ax.bar(x - 0.2, g["ultralytics.iid_test.mAP50"] * 100, 0.4, label="IID test", color="#2a6f97")
    ax.bar(x + 0.2, g["ultralytics.ood.mAP50"] * 100, 0.4, label="OOD", color="#e07a1f")
    ax.set_xticks(x, labels, fontsize=8); ax.set_ylabel("mAP50 (%)"); ax.set_ylim(0, 100)
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(rdir / "fig_ood_map50.png", dpi=200); plt.close(fig)

    cols = ["coco.ood.AP_small", "coco.ood.AP_medium", "coco.ood.AP_large"]
    if all(c in g for c in cols):
        fig, ax = plt.subplots(figsize=(max(6, len(g) * 0.9), 3.6))
        for i, (c, col) in enumerate(zip(cols, ["#2a6f97", "#61a5c2", "#a9d6e5"])):
            ax.bar(x + (i - 1) * 0.27, g[c] * 100, 0.27, label=c.split("_")[-1], color=col)
        ax.set_xticks(x, labels, fontsize=8); ax.set_ylabel("OOD AP 50:95 (%)")
        ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(rdir / "fig_size_ap.png", dpi=200); plt.close(fig)

    if seq_cols:
        fig, ax = plt.subplots(figsize=(7, 3.6))
        seqs = [c.split(".")[1] for c in seq_cols]
        for i, lab in enumerate(labels):
            ax.plot(seqs, [g.iloc[i][c] * 100 for c in seq_cols], marker="o",
                    label=lab.replace("\n", " "))
        ax.set_ylabel("OOD AP50 (%)"); ax.legend(fontsize=7, frameon=False, ncol=2)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(rdir / "fig_per_sequence.png", dpi=200); plt.close(fig)
    print(f"figures written to {rdir}")


if __name__ == "__main__":
    main()

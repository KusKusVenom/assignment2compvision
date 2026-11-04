"""Build results table, training curves, confusion matrix and Grad-CAM figure from runs/."""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from common import get_device, get_hard_test, grad_cam, load_mnist, load_model

os.makedirs("figures", exist_ok=True)
runs = {}
for p in sorted(glob.glob("runs/*/metrics.json")):
    m = json.load(open(p))
    runs[m["name"]] = m
assert runs, "No runs found - run ./run_all.sh first"

# ---- results table -------------------------------------------------------
rows = []
for n, m in runs.items():
    rows.append({
        "Run": n, "Model": "ConvNeXt-T (pretrained)" if m["model"] == "sota" else "Small CNN (baseline)",
        "Synthetic share": f"{int(m['synth_frac'] * 100)}%", "Params (M)": round(m["params_M"], 2),
        "Clean Acc": round(m["clean"]["acc"], 4), "Clean macro-F1": round(m["clean"]["macro_f1"], 4),
        "Hard Acc": round(m["hard"]["acc"], 4), "Hard macro-F1": round(m["hard"]["macro_f1"], 4),
        "Latency (ms/img)": round(m["latency_ms"], 2),
    })
df = pd.DataFrame(rows)
df.to_csv("figures/results_table.csv", index=False)
try:
    md = df.to_markdown(index=False)
except ImportError:
    md = df.to_string(index=False)
open("figures/results_table.md", "w").write(md)
print(df.to_string(index=False))

# ---- training curves -----------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
for n in runs:
    c = pd.read_csv(f"runs/{n}/curves.csv")
    ax[0].plot(c.epoch, c.train_loss, marker="o", label=n)
    ax[1].plot(c.epoch, c.val_loss, marker="o", label=n)
    ax[2].plot(c.epoch, c.val_acc, marker="o", label=n)
for a, t in zip(ax, ["Train loss", "Validation loss", "Validation accuracy"]):
    a.set_title(t); a.set_xlabel("epoch"); a.grid(alpha=.3)
ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig("figures/training_curves.png", dpi=150); plt.close()

# ---- confusion matrix on the hard test set (best SOTA run) ---------------
best = max(runs.values(), key=lambda m: m["hard"]["acc"])
cm = np.array(best["hard"]["confusion"])
plt.figure(figsize=(5, 4.5)); plt.imshow(cm, cmap="Blues")
plt.title(f"Hard test confusion - {best['name']}"); plt.xlabel("predicted"); plt.ylabel("true")
plt.xticks(range(10)); plt.yticks(range(10)); plt.colorbar()
plt.tight_layout(); plt.savefig("figures/confusion_hard.png", dpi=150); plt.close()

# ---- Grad-CAM comparison on hard test images -----------------------------
device = get_device()
(_, _), (_, _), (xte, yte) = load_mnist()
xhard = get_hard_test(xte)
names = [n for n in ["baseline_synth0", "sota_synth50"] if n in runs]
models = {n: load_model(runs[n]["model"], f"runs/{n}/model.pt", device) for n in names}
idxs = list(range(8))
fig, ax = plt.subplots(1 + len(names), len(idxs), figsize=(2 * len(idxs), 2.2 * (1 + len(names))))
for j, i in enumerate(idxs):
    ax[0, j].imshow(xhard[i, 0], cmap="gray"); ax[0, j].set_title(f"true {int(yte[i])}", fontsize=8)
    for r, n in enumerate(names, start=1):
        cam, pred, probs = grad_cam(models[n], xhard[i:i + 1].to(device))
        ax[r, j].imshow(xhard[i, 0], cmap="gray"); ax[r, j].imshow(cam, cmap="jet", alpha=0.5)
        ax[r, j].set_title(f"{n.split('_')[0]}: {pred} ({probs[pred]:.0%})", fontsize=7)
for a in ax.ravel():
    a.axis("off")
plt.tight_layout(); plt.savefig("figures/gradcam_comparison.png", dpi=150); plt.close()
print("Saved figures/ (results_table, training_curves, confusion_hard, gradcam_comparison)")

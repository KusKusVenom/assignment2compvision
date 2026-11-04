"""Train one configuration and save metrics, curves and weights.

  python train.py --model baseline --synth 0
  python train.py --model sota     --synth 0.5
"""
import argparse
import csv
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import (DigitDataset, bench_latency, build_model, compute_metrics,
                    get_device, get_hard_test, load_mnist, predict_logits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["baseline", "sota"], required=True)
    ap.add_argument("--synth", type=float, default=0.0, help="share of training samples corrupted on the fly")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()

    epochs = a.epochs or (5 if a.model == "baseline" else 3)
    lr = a.lr or (1e-3 if a.model == "baseline" else 2e-4)
    name = f"{a.model}_synth{int(a.synth * 100)}"
    out = os.path.join(a.out, name)
    os.makedirs(out, exist_ok=True)

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = get_device()
    print(f"== {name} | device={device} | epochs={epochs} lr={lr}")

    (xtr, ytr), (xva, yva), (xte, yte) = load_mnist()
    xhard = get_hard_test(xte)
    loader = DataLoader(DigitDataset(xtr, ytr, a.synth), batch_size=a.bs, shuffle=True,
                        num_workers=2, drop_last=True)

    model = build_model(a.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader))
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    curves = []
    for ep in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(xb); n += len(xb)
        v = compute_metrics(predict_logits(model, xva, device), yva)
        curves.append({"epoch": ep, "train_loss": tot / n, "val_loss": v["loss"], "val_acc": v["acc"]})
        print(f"epoch {ep}: train_loss={tot / n:.4f} val_loss={v['loss']:.4f} val_acc={v['acc']:.4f}")

    clean = compute_metrics(predict_logits(model, xte, device), yte)
    hard = compute_metrics(predict_logits(model, xhard, device), yte)
    latency = bench_latency(model, device)

    metrics = {
        "name": name, "model": a.model, "synth_frac": a.synth, "epochs": epochs, "lr": lr,
        "params_M": n_params / 1e6, "latency_ms": latency,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "clean": clean, "hard": hard,
    }
    json.dump(metrics, open(os.path.join(out, "metrics.json"), "w"), indent=2)
    with open(os.path.join(out, "curves.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(curves[0])); w.writeheader(); w.writerows(curves)
    torch.save(model.state_dict(), os.path.join(out, "model.pt"))
    print(f"clean acc={clean['acc']:.4f} F1={clean['macro_f1']:.4f} | "
          f"hard acc={hard['acc']:.4f} F1={hard['macro_f1']:.4f} | {latency:.2f} ms/img")


if __name__ == "__main__":
    main()

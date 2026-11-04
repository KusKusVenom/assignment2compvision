"""Gradio demo: draw/upload a digit -> prediction + Grad-CAM + latency + 12 hard variants."""
import argparse
import os
import platform
import random
import time

import gradio as gr
import matplotlib
import numpy as np
import torch
from PIL import Image

from common import bench_latency, get_device, grad_cam, hard_transform, load_model

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="sota_synth50", help="folder name inside runs/")
ap.add_argument("--share", action="store_true")
args, _ = ap.parse_known_args()

DEVICE = get_device()
run_dir = os.path.join("runs", args.run)
MODEL_NAME = "sota" if args.run.startswith("sota") else "baseline"
MODEL = load_model(MODEL_NAME, os.path.join(run_dir, "model.pt"), DEVICE)
LATENCY = bench_latency(MODEL, DEVICE, n=30)
ENV = f"{'GPU ' + torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU'} | {platform.platform()} | torch {torch.__version__}"
CMAP = matplotlib.colormaps["jet"]


def preprocess(img):
    """Any drawing/photo -> 28x28 float array, white digit on black (MNIST style)."""
    if isinstance(img, dict):
        img = img.get("composite")
    if img is None:
        return None
    arr = np.array(img)
    if arr.ndim == 2:
        lum = arr.astype(float)
    elif arr.shape[2] == 4:                                   # RGBA -> composite on white
        al = arr[..., 3:4] / 255.0
        lum = (arr[..., :3] * al + 255 * (1 - al)).mean(2)
    else:
        lum = arr[..., :3].astype(float).mean(2)
    x = 1 - lum / 255.0
    if x.mean() > 0.5:                                        # background should be dark
        x = 1 - x
    lo, hi = np.percentile(x, 5), np.percentile(x, 99.5)
    x = np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)
    ys, xs = np.where(x > 0.3)
    if len(ys) < 10:
        return None
    x = x[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    s = max(x.shape)
    sq = np.zeros((s, s), dtype=np.float32)
    sq[(s - x.shape[0]) // 2:(s - x.shape[0]) // 2 + x.shape[0],
       (s - x.shape[1]) // 2:(s - x.shape[1]) // 2 + x.shape[1]] = x
    small = Image.fromarray((sq * 255).astype(np.uint8)).resize((20, 20), Image.LANCZOS)
    canvas = np.zeros((28, 28), dtype=np.float32)
    canvas[4:24, 4:24] = np.array(small) / 255.0
    return canvas


def overlay(x28, cam):
    base = np.array(Image.fromarray((x28 * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)) / 255.0
    cam_big = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)) / 255.0
    heat = CMAP(cam_big)[..., :3]
    out = 0.5 * np.stack([base] * 3, -1) + 0.5 * heat
    return (out * 255).astype(np.uint8)


def predict(img):
    x = preprocess(img)
    if x is None:
        return None, None, None, "Draw or upload a digit first (dark ink on a light background)."
    t = torch.from_numpy(x)[None, None].to(DEVICE)
    cam, idx, probs = grad_cam(MODEL, t)
    label = {str(i): float(p) for i, p in enumerate(probs)}
    info = (f"Prediction: **{idx}** ({probs[idx]:.1%})  \n"
            f"Avg inference latency: **{LATENCY:.2f} ms/img** (batch 1)  \n"
            f"Model: `{args.run}` | Environment: {ENV}")
    proc = np.array(Image.fromarray((x * 255).astype(np.uint8)).resize((224, 224), Image.NEAREST))
    return label, overlay(x, cam), proc, info


def variants(sketch_img, upload_img):
    x = preprocess(upload_img)
    if x is None:
        x = preprocess(sketch_img)
    if x is None:
        return []
    base = torch.from_numpy(x)[None]
    items = []
    for _ in range(12):
        v = hard_transform(base.clone())
        with torch.no_grad():
            p = MODEL(v[None].to(DEVICE)).softmax(1)[0]
        k = int(p.argmax())
        im = Image.fromarray((v[0].numpy() * 255).astype(np.uint8)).resize((112, 112), Image.NEAREST)
        items.append((np.array(im), f"pred {k} ({p[k]:.0%})  [SYNTHETIC]"))
    return items


def make_sketchpad():
    try:
        brush = gr.Brush(default_size=14, colors=["#000000"], default_color="#000000", color_mode="fixed")
        return gr.Sketchpad(label="Draw a digit", canvas_size=(280, 280), brush=brush)
    except Exception:
        return gr.Sketchpad(label="Draw a digit")


with gr.Blocks(title="Handwritten digits - ConvNeXt + Grad-CAM") as demo:
    gr.Markdown("# Handwritten digit recognition (A2: SOTA upgrade)\n"
                "Pretrained **ConvNeXt-Tiny** fine-tuned on MNIST with synthetic hard examples. "
                "Grad-CAM shows which pixels drove the decision.\n\n"
                "> **Notice:** the *variants* gallery below is **synthetic / generated** media created by "
                "procedural corruptions (thin strokes, ink bleed, blur, noise, rotation). No faces or personal data are used.")
    with gr.Row():
        with gr.Column():
            with gr.Tab("Draw"):
                sketch = make_sketchpad()
                b1 = gr.Button("Predict drawing", variant="primary")
            with gr.Tab("Upload / camera"):
                up = gr.Image(label="Photo of a digit", sources=["upload", "webcam"], type="numpy")
                b2 = gr.Button("Predict upload", variant="primary")
        with gr.Column():
            lab = gr.Label(num_top_classes=3, label="Prediction")
            cam = gr.Image(label="Grad-CAM (red = most important)", height=224)
            proc = gr.Image(label="What the model sees (28x28, centred)", height=224)
            info = gr.Markdown()
    gr.Markdown("### 12 synthetic hard variants of your digit")
    vb = gr.Button("Generate 12 variants")
    gal = gr.Gallery(columns=6, height=260, label="Synthetic variants + model prediction")
    b1.click(predict, sketch, [lab, cam, proc, info])
    b2.click(predict, up, [lab, cam, proc, info])
    vb.click(variants, [sketch, up], gal)

if __name__ == "__main__":
    demo.launch(share=args.share)

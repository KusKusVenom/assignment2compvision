"""Shared code: data, hard-example synthesis, models, Grad-CAM, metrics."""
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchvision import datasets

DATA_ROOT = "data"
HARD_TEST_PATH = "data/hard_test_seed123.pt"


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# Hard-example synthesis (procedural): thin strokes, marker bleed, blur,
# noise, rotation, shift/scale.  Input/Output: 1xHxW float tensor in [0, 1].
# --------------------------------------------------------------------------
def thin_stroke(x):
    return -F.max_pool2d(-x[None], 3, 1, 1)[0]            # erosion


def marker_bleed(x):
    y = F.max_pool2d(x[None], 3, 1, 1)                     # dilation
    return F.avg_pool2d(y, 3, 1, 1)[0]                     # ink spreading


def blur(x):
    s = random.uniform(0.8, 1.6)
    return TF.gaussian_blur(x, kernel_size=[5, 5], sigma=[s, s])


def noise(x):
    return (x + 0.15 * torch.randn_like(x)).clamp(0, 1)


def rotate(x):
    return TF.rotate(x, random.uniform(-35, 35),
                     interpolation=TF.InterpolationMode.BILINEAR)


def shift_scale(x):
    return TF.affine(x, angle=0.0,
                     translate=[random.randint(-3, 3), random.randint(-3, 3)],
                     scale=random.uniform(0.7, 1.2), shear=[0.0, 0.0],
                     interpolation=TF.InterpolationMode.BILINEAR)


HARD_OPS = [thin_stroke, marker_bleed, blur, noise, rotate, shift_scale]


def hard_transform(x):
    """Apply 1-3 random corruptions to a 1x28x28 tensor."""
    for op in random.sample(HARD_OPS, random.choice([1, 2, 3])):
        x = op(x)
    return x.clamp(0, 1)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_mnist(root=DATA_ROOT, val_size=5000, seed=0):
    tr = datasets.MNIST(root, train=True, download=True)
    te = datasets.MNIST(root, train=False, download=True)
    x = tr.data.float().div(255).unsqueeze(1)
    y = tr.targets.clone()
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(x), generator=g)
    v, t = perm[:val_size], perm[val_size:]
    x_te = te.data.float().div(255).unsqueeze(1)
    return (x[t], y[t]), (x[v], y[v]), (x_te, te.targets.clone())


def get_hard_test(x_test, seed=123, path=HARD_TEST_PATH):
    """Fixed corrupted copy of the test set (same for every run)."""
    import os
    if os.path.exists(path):
        return torch.load(path)
    random.seed(seed)
    torch.manual_seed(seed)
    out = torch.stack([hard_transform(xi) for xi in x_test])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(out, path)
    return out


class DigitDataset(Dataset):
    """Training set where a fraction `synth_frac` of samples is corrupted on the fly."""

    def __init__(self, x, y, synth_frac=0.0):
        self.x, self.y, self.p = x, y, synth_frac

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        xi = self.x[i]
        if self.p > 0 and random.random() < self.p:
            xi = hard_transform(xi)
        return xi, self.y[i]


# --------------------------------------------------------------------------
# Models (all take Bx1x28x28 in [0,1])
# --------------------------------------------------------------------------
class BaselineCNN(nn.Module):
    """Small CNN used as the 'A1 baseline'."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.act2 = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.act3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.drop(self.pool(x)).flatten(1)
        return self.fc2(self.act3(self.fc1(x)))

    def cam_layer(self):
        return self.act2


class SotaNet(nn.Module):
    """ImageNet-pretrained ConvNeXt-Tiny fine-tuned on digits."""

    def __init__(self, pretrained=True, size=128):
        super().__init__()
        import timm
        self.net = timm.create_model("convnext_tiny", pretrained=pretrained, num_classes=10)
        self.size = size
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        x = x.expand(-1, 3, -1, -1)
        return self.net((x - self.mean) / self.std)

    def cam_layer(self):
        return self.net.stages[-1]


def build_model(name, pretrained=True):
    return BaselineCNN() if name == "baseline" else SotaNet(pretrained=pretrained)


def load_model(name, weights, device):
    m = build_model(name, pretrained=False)
    m.load_state_dict(torch.load(weights, map_location="cpu"))
    return m.to(device).eval()


# --------------------------------------------------------------------------
# Inference helpers, metrics, latency, Grad-CAM
# --------------------------------------------------------------------------
@torch.no_grad()
def predict_logits(model, X, device, bs=512):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = X[i:i + bs].to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out.append(model(xb).float().cpu())
    return torch.cat(out)


def compute_metrics(logits, y):
    from sklearn.metrics import confusion_matrix, f1_score
    pred = logits.argmax(1).numpy()
    yt = y.numpy()
    return {
        "loss": float(F.cross_entropy(logits, y)),
        "acc": float((pred == yt).mean()),
        "macro_f1": float(f1_score(yt, pred, average="macro")),
        "confusion": confusion_matrix(yt, pred, labels=list(range(10))).tolist(),
    }


@torch.no_grad()
def bench_latency(model, device, n=100, warmup=10):
    """Average single-image forward time in milliseconds."""
    model.eval()
    x = torch.rand(1, 1, 28, 28, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def grad_cam(model, x, class_idx=None):
    """x: 1x1x28x28 tensor on the model's device. Returns (cam 28x28 in [0,1], class, probs)."""
    model.eval()
    layer = model.cam_layer()
    store = {}

    def fwd_hook(_, __, out):
        store["a"] = out
        out.register_hook(lambda g: store.__setitem__("g", g))

    h = layer.register_forward_hook(fwd_hook)
    try:
        x = x.clone().requires_grad_(True)
        logits = model(x.float())
        probs = logits.softmax(1)
        idx = int(logits.argmax(1)) if class_idx is None else class_idx
        model.zero_grad()
        logits[0, idx].backward()
    finally:
        h.remove()
    a, g = store["a"], store["g"]
    w = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy(), idx, probs[0].detach().cpu().numpy()

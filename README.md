# Handwritten digits - A2: SOTA upgrade + demo

**Topic:** handwritten digit recognition (MNIST).
**Baseline (A1 stand-in):** small 2-conv CNN.
**SOTA component:** ImageNet-pretrained **ConvNeXt-Tiny** (timm) fine-tuned on digits.
**Data synthesis:** procedural hard examples (thin stroke, marker bleed, blur, noise, rotation, shift/scale) applied on the fly to 50% of training samples.
**Explainability:** Grad-CAM. **App:** Gradio (draw / upload -> prediction + Grad-CAM + latency + 12 synthetic variants).

## Run (Google Colab, easiest)
Open `run_in_colab.ipynb`, choose a T4 GPU runtime, run the cells.

## Run (local, GPU recommended)
```bash
pip install -r requirements.txt tabulate
bash run_all.sh                     # trains 4 runs -> runs/*, makes figures/*
python app.py --run sota_synth50    # add --share for a public link
```

## Experiment design (2x2)
| Run | Model | Synthetic share |
|---|---|---|
| baseline_synth0 | small CNN | 0% (A1 baseline) |
| baseline_synth50 | small CNN | 50% |
| sota_synth0 | ConvNeXt-T | 0% |
| sota_synth50 | ConvNeXt-T | 50% (full method) |

Metrics: accuracy + macro-F1 on the **clean** MNIST test set and on a **hard** test set (the same test images corrupted once with a fixed seed, so every run sees identical hard images), single-image latency, training curves (validation = 5k held-out train images).

## Files
`common.py` data, corruptions, models, Grad-CAM, metrics - `train.py` one run - `report_figs.py` table + figures - `app.py` demo - `run_all.sh` everything.

## Notes
- Fixed seed 0; hard test seed 123. Results vary slightly by GPU.
- Synthetic images are procedurally generated and labelled **SYNTHETIC** in the app. No faces or personal data are used.

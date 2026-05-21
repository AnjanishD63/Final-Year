# Final-Year · Multimodal Emotion Recognition

PyTorch fused model that combines:

- Transformer text embeddings (BERT → 1536‑D pooled vector)
- 40‑D MFCC statistics for uploaded / recorded audio clips
- 48×48 grayscale face-appearance descriptors from camera frames

Predictions target four classes (`Angry`, `Happy`, `Sad`, `Neutral`). The bundled pipeline ships with a **deterministic synthetic IEMOCAP-style tensor bundle** (`./features/*.npy`) purely so cloning the repo succeeds without external corpora—you should replace those artifacts with genuinely extracted multimodal embeddings for research-grade fidelity.

---

## One-command launcher

Run everything (dependency install → synthetic corpus bake → fused training if needed → corpus evaluation UI boot):

```bash
python run_project.py
```

Useful overrides:

```bash
# Skip pip stage if you manage environments manually
python run_project.py --skip-pip

# Regenerate randomized tensors before training
python run_project.py --fresh-data --retrain

# Already trained everything? Jump straight into Streamlit
python run_project.py --skip-pip --launch-ui-only
```

Tune training without editing Python:

```bash
EMOTION_EPOCHS=8 EMOTION_PATIENCE=3 EMOTION_BATCH=24 python run_project.py --retrain --skip-pip
```

---

### Component map

| File | Responsibility |
| --- | --- |
| `run_project.py` | End-to-end bootstrap + optional training + launches `app.py` |
| `demo_features.py` | Creates normalized `.npy` tensors + saves `features/normalization.npz` statistics |
| `train_runner.py` | Lion optimizer + cosine schedule + gated fusion classifier |
| `model_def.py` | `FusionEmotionModel` declaration |
| `preprocessing.py` | Shared modality extractors aligned with demos |
| `app.py` | Streamlit multimodal UI |
| `realtime_application_new.py` | OpenCV desktop loop (sentence + WAV + live frame inputs) |

---

## Manual commands (optional)

Train only (`best_model_tav.pt` lands in repo root):

```bash
python train_new.py
```

Evaluate only:

```bash
python test_new.py
```

Web UI without orchestrator extras:

```bash
streamlit run app.py
```

---

> **Realtime desktop loop:** Needs a GUI-enabled OpenCV build (`opencv-python`, not `-headless`) because it calls `cv2.imshow`. Streamlit + background inference tolerate headless installs.

## Torch wheels

CPU wheels install by default via `requirements.txt`. For NVIDIA-backed builds, reinstall PyTorch from <https://pytorch.org> after deps finish installing.

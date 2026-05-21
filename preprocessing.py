"""Feature extraction helpers shared by training demo data generation, inference, and Streamlit."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Tuple

import cv2
import librosa
import numpy as np
import torch
from transformers import BertModel, BertTokenizer

EMOTION_LABELS = ("Angry", "Happy", "Sad", "Neutral")

FRAME_SIZE = 48


def normalize_feature_matrix(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, dtype=np.float64))
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)


def normalize_with_stats_row(x_row: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Apply fixed training/demo statistics along the feature dimension (single row accepted)."""
    x = np.nan_to_num(np.asarray(x_row, dtype=np.float64))
    denom = sigma.astype(np.float64) + 1e-6
    out = ((x - mu.astype(np.float64)) / denom).astype(np.float32)
    return out.reshape(1, -1)


def load_normalization_arrays(repo_root: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load (mu, sigma pairs) aligned with modality dimensions from features/normalization.npz.
    Returns (text_mu,text_sigma,audio_mu,audio_sigma,video_mu,video_sigma).
    """
    root = Path(repo_root).resolve()
    path = root / "features" / "normalization.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run demo feature generation first (via run_project.py or demo_features)."
        )
    blob = np.load(path)
    return (
        blob["text_mu"],
        blob["text_sigma"],
        blob["audio_mu"],
        blob["audio_sigma"],
        blob["video_mu"],
        blob["video_sigma"],
    )


def _lazy_bert(device: torch.device):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert = BertModel.from_pretrained("bert-base-uncased").to(device)
    bert.eval()
    return tokenizer, bert


@torch.no_grad()
def embed_text_batch(
    sentences: list[str],
    device: torch.device,
    tokenizer: BertTokenizer | None = None,
    bert: BertModel | None = None,
) -> np.ndarray:
    """1536-D text vectors: concatenate two [CLS]-style pooled 768-D embeddings."""
    tok, lm = tokenizer, bert
    if tok is None or lm is None:
        tok, lm = _lazy_bert(device)
    feats: list[np.ndarray] = []
    for line in sentences:
        enc = tok(
            line,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        inp = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        out = lm(input_ids=inp, attention_mask=mask)
        hidden = out.last_hidden_state[:, 0, :]  # [1,768]
        h = hidden.squeeze(0).cpu().numpy().astype(np.float32)
        emb = np.concatenate([h, h], axis=-1).astype(np.float32)
        feats.append(emb)
    return np.stack(feats, axis=0)


def mfcc_from_audio(samples: np.ndarray, sr: int) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=samples.astype(np.float32), sr=int(sr), n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1).astype(np.float32)
    return mfcc_mean


def audio_bytes_to_feature(audio_bytes: bytes, sr: int = 16000) -> np.ndarray:
    samples, _ = librosa.load(io.BytesIO(audio_bytes), sr=sr, mono=True)
    return mfcc_from_audio(samples, sr)


def video_bgr_to_feature(frame_bgr: np.ndarray | None, face_ok: bool = True) -> np.ndarray:
    """
    Produce a single 2304-D vector aligned with demos: resized grayscale face/global region.
    Mirrors realtime_application logic (mean across repeated tiles is same as flattened patch).
    """
    if frame_bgr is None or frame_bgr.size == 0:
        gray = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.float32)
    else:
        gray_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        faces = ()
        if face_ok:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            faces = face_cascade.detectMultiScale(gray_full, 1.3, 5)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_roi = gray_full[y : y + h, x : x + w]
        else:
            face_roi = gray_full

        gray = cv2.resize(face_roi, (FRAME_SIZE, FRAME_SIZE)).astype(np.float32) / 255.0

    return gray.flatten().astype(np.float32)


def tensors_for_model(
    text: str,
    audio_feature_40: np.ndarray,
    video_vec_2304: np.ndarray,
    device: torch.device,
    stats: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tokenizer: BertTokenizer | None = None,
    bert: BertModel | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    stats = (mu_t, sg_t, mu_a, sg_a, mu_v, sg_v), each aligned to modality feature dim,
    computed from the demo/training corpus (saved in features/normalization.npz).
    """
    mu_t, sg_t, mu_a, sg_a, mu_v, sg_v = stats

    raw_txt = embed_text_batch([text], device, tokenizer, bert)[0].reshape(-1).astype(np.float32)
    t_np = normalize_with_stats_row(raw_txt, mu_t, sg_t)

    raw_a = np.asarray(audio_feature_40.reshape(-1), dtype=np.float32)
    a_np = normalize_with_stats_row(raw_a, mu_a, sg_a)

    raw_v = np.asarray(video_vec_2304.reshape(-1), dtype=np.float32)
    v_np = normalize_with_stats_row(raw_v, mu_v, sg_v)

    t = torch.from_numpy(t_np).to(device)
    a = torch.from_numpy(a_np).to(device)
    v = torch.from_numpy(v_np).to(device)
    return t, a, v

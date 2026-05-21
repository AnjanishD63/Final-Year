"""Create a reproducible multimodal demo dataset under ./features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from preprocessing import FRAME_SIZE, embed_text_batch, mfcc_from_audio


def _patterns_for_demo(n: int, rng: np.random.Generator) -> tuple[list[str], list[int]]:
    """Synthetic sentences loosely themed by label (demo coherence only)."""
    buckets = (
        ["I hate this!", "Leave me alone.", "Stop doing that!", "This is unbearable!", "Enough already!"],  # Angry
        ["I feel amazing today!", "What a lovely surprise!", "I am so cheerful!", "Great news everyone!", "I love this vibe!"],  # Happy
        ["Everything feels hopeless.", "I miss them so much.", "I can't stop crying.", "It hurts every day.", "I feel broken inside."],  # Sad
        ["Nothing much happened.", "It was an ordinary meeting.", "I have no opinion really.", "The weather was fine.", "Just another weekday."],  # Neutral
    )

    sentences: list[str] = []
    labels: list[int] = []
    for i in range(n):
        lab = i % 4
        pool = buckets[lab]
        phrase = rng.choice(pool)
        sentences.append(f"{phrase} [sample:{i}]")
        labels.append(lab)
    return sentences, labels


def _demo_audio_wave(i: int, sr: int = 16000, seconds: float = 2.0) -> tuple[np.ndarray, int]:
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False).astype(np.float32)
    f0 = 180.0 + (i % 50) * 3.14159
    mod = np.sin(t * (2 + 0.01 * i)) * 0.08
    y = np.sin((2 * np.pi * f0 / sr) * t + mod).astype(np.float32)
    y += rng_like(i, scale=0.01, n=len(y))
    return y, sr


def rng_like(seed: int, scale: float, n: int) -> np.ndarray:
    generator = np.random.default_rng((seed + 1337) & 0xFFFFFFFF)
    return generator.standard_normal(size=n).astype(np.float32) * scale


def _demo_face_vector(i: int, label: int) -> np.ndarray:
    """Controlled synthetic 2304-D grayscale patch (48 × 48) flattened."""
    generator = np.random.default_rng((i * 17 + label * 31) & 0xFFFFFFFF)
    face = generator.random((FRAME_SIZE, FRAME_SIZE), dtype=np.float32)
    gx, gy = np.meshgrid(np.linspace(0, 1, FRAME_SIZE), np.linspace(0, 1, FRAME_SIZE))
    swirl = np.sin(gx * 6 + label + i * 0.01) * np.cos(gy * 5 + label * 0.5)
    face = face + 0.25 * swirl.astype(np.float32)
    face = np.clip(face, 0.0, 1.0)
    return face.reshape(-1).astype(np.float32)


def build_demo_feature_pack(
    repo_root: Path,
    *,
    num_samples: int = 520,
    device: torch.device | None = None,
    regenerate: bool = False,
    seed_text: int = 42,
) -> Path:
    """
    Writes normalized tensors plus statistics reusable at inference:

    - features/text_bert.npy
    - features/audio_wav2vec.npy
    - features/video_processed.npy
    - features/labels.npy
    - features/normalization.npz (raw μ / σ mirrors used for live inference).
    """

    feat_dir = repo_root / "features"
    markers = (feat_dir / "labels.npy", feat_dir / "normalization.npz")
    if all(path.exists() for path in markers) and not regenerate:
        return feat_dir

    rng = np.random.default_rng(seed_text)
    feat_dir.mkdir(parents=True, exist_ok=True)

    development_device = torch.device("cpu") if device is None else device
    sentences, label_list = _patterns_for_demo(num_samples, rng)
    labels_np = np.array(label_list, dtype=np.int64)

    text_matrix = embed_text_batch(sentences, development_device).astype(np.float32)

    audio_feats: list[np.ndarray] = []
    video_feats: list[np.ndarray] = []
    for idx in range(num_samples):
        wave, waveform_sr = _demo_audio_wave(idx, sr=16000)
        audio_feats.append(mfcc_from_audio(wave, waveform_sr))
        video_feats.append(_demo_face_vector(idx, int(labels_np[idx])))

    audio_stack = np.stack(audio_feats).astype(np.float32)
    video_stack = np.stack(video_feats).astype(np.float32)
    text_raw = np.nan_to_num(text_matrix)

    def moments(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sane = np.nan_to_num(matrix)
        mu_axis = sane.mean(axis=0)
        sigma_axis = sane.std(axis=0) + 1e-6
        return mu_axis.astype(np.float32), sigma_axis.astype(np.float32)

    t_mu, t_sigma = moments(text_raw)
    a_mu, a_sigma = moments(audio_stack)
    v_mu, v_sigma = moments(video_stack)

    text_norm = ((text_raw - t_mu) / (t_sigma.astype(np.float64) + 1e-6)).astype(np.float32)
    audio_norm = ((np.nan_to_num(audio_stack) - a_mu) / (a_sigma.astype(np.float64) + 1e-6)).astype(np.float32)
    video_norm = ((np.nan_to_num(video_stack) - v_mu) / (v_sigma.astype(np.float64) + 1e-6)).astype(np.float32)

    np.save(feat_dir / "text_bert.npy", text_norm)
    np.save(feat_dir / "audio_wav2vec.npy", audio_norm)
    np.save(feat_dir / "video_processed.npy", video_norm)
    np.save(feat_dir / "labels.npy", labels_np)

    np.savez(
        feat_dir / "normalization.npz",
        text_mu=t_mu,
        text_sigma=t_sigma,
        audio_mu=a_mu,
        audio_sigma=a_sigma,
        video_mu=v_mu,
        video_sigma=v_sigma,
    )

    print(f"[demo-features] Synthetic corpus saved ({num_samples} rows) → {feat_dir}")

    return feat_dir

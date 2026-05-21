"""Train / evaluate FusionEmotionModel on prepared feature tensors under ./features."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset, random_split

from model_def import FusionEmotionModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Lion(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 1e-4,
    ):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]

                p.mul_(1 - group["lr"] * group["weight_decay"])

                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group["lr"])

                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        log_probs = torch.log_softmax(pred, dim=1)
        n_classes = pred.size(1)
        smooth = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
        smooth.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
        return -(smooth * log_probs).sum(dim=1).mean()


class FeatureDataset(Dataset):
    """Loads precomputed, already-normalized per-modality tensors from disk."""

    def __init__(self, features_dir: Path):
        fd = Path(features_dir)
        self.t = np.load(fd / "text_bert.npy")
        self.a = np.load(fd / "audio_wav2vec.npy")
        self.v = np.load(fd / "video_processed.npy")
        self.y = np.load(fd / "labels.npy")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.t[i], dtype=torch.float32),
            torch.tensor(self.a[i], dtype=torch.float32),
            torch.tensor(self.v[i], dtype=torch.float32),
            torch.tensor(self.y[i], dtype=torch.long),
        )


def train_fusion(repo_root: str | Path, *, epochs: int | None = None, patience: int | None = None) -> Path:
    root = Path(repo_root).resolve()
    feat_dir = root / "features"
    ckpt = root / "best_model_tav.pt"

    ep = epochs if epochs is not None else int(os.environ.get("EMOTION_EPOCHS", "14"))
    pat = patience if patience is not None else int(os.environ.get("EMOTION_PATIENCE", "5"))

    batch_size = int(os.environ.get("EMOTION_BATCH", "48"))
    lr = float(os.environ.get("EMOTION_LR", str(3e-5)))

    dataset = FeatureDataset(feat_dir)
    train_sz = max(2, int(0.8 * len(dataset)))
    val_sz = len(dataset) - train_sz

    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset, [train_sz, val_sz], generator=generator)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin, num_workers=0)

    model = FusionEmotionModel().to(DEVICE)
    optim = Lion(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=ep)
    loss_fn = LabelSmoothingLoss()

    print(f"[train] Device={DEVICE} epochs={ep} batch={batch_size} samples_total={len(dataset)}")

    best = 0.0
    no_imp = 0

    for epoch in range(ep):
        model.train()
        correct = total = 0
        for t, a, v, y in train_loader:
            t, a, v, y = t.to(DEVICE), a.to(DEVICE), v.to(DEVICE), y.to(DEVICE)
            optim.zero_grad()
            out = model(t, a, v)
            loss = loss_fn(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        train_acc = correct / max(1, total) * 100

        model.eval()
        eval_correct = eval_total = 0
        with torch.no_grad():
            for t, a, v, y in val_loader:
                t, a, v, y = t.to(DEVICE), a.to(DEVICE), v.to(DEVICE), y.to(DEVICE)
                out = model(t, a, v)
                eval_correct += (out.argmax(1) == y).sum().item()
                eval_total += y.size(0)

        val_acc = eval_correct / max(1, eval_total) * 100

        sched.step()

        print(f"[train] Epoch {epoch+1}/{ep} train_acc={train_acc:.1f}% val_acc={val_acc:.1f}%")

        if val_acc > best:
            best = val_acc
            no_imp = 0
            torch.save(model.state_dict(), ckpt)
        else:
            no_imp += 1

        if no_imp >= pat:
            print(f"[train] Early stopping ({pat} stagnant epochs)")
            break

    print(f"[train] Best validation accuracy≈ {best:.2f}% checkpoint → {ckpt}")
    return ckpt


def evaluate_fusion(repo_root: str | Path, *, ckpt_rel: str = "best_model_tav.pt", batch_size: int = 8) -> None:
    root = Path(repo_root).resolve()
    fd = root / "features"
    ckpt = root / ckpt_rel

    ds = FeatureDataset(fd)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = FusionEmotionModel().to(DEVICE)
    blob = ckpt.resolve()
    if not blob.exists():
        raise FileNotFoundError(blob)
    state = torch.load(blob, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    preds: list[int] = []
    labels: list[int] = []

    with torch.no_grad():
        for t, a, v, y in loader:
            out = model(t.to(DEVICE), a.to(DEVICE), v.to(DEVICE))
            p = out.argmax(1).cpu().numpy().tolist()
            preds.extend([int(i) for i in p])
            labels.extend(y.numpy().astype(int).tolist())

    names = ["Angry", "Happy", "Sad", "Neutral"]

    report = classification_report(labels, preds, labels=[0, 1, 2, 3], target_names=names, zero_division=0)

    acc = np.mean(np.array(preds) == np.array(labels)) * 100.0 if labels else 0.0

    print("\n[eval] Metrics below use the saved feature matrix plus the fused checkpoint:")
    print(f"[eval] Whole-corpus accuracy≈ {acc:.2f}%\n")
    print(report)

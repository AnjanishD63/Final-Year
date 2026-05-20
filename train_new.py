import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 32
EPOCHS = 20
LR = 3e-5


# ================= LION =================
class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=1e-4):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                p.mul_(1 - group['lr'] * group['weight_decay'])

                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group['lr'])

                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)


# ================= LABEL SMOOTHING =================
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        log_probs = torch.log_softmax(pred, dim=1)
        n_classes = pred.size(1)

        smooth = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
        smooth.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)

        return -(smooth * log_probs).sum(dim=1).mean()


loss_fn = LabelSmoothingLoss()


# ================= DATA =================
def normalize(x):
    x = np.nan_to_num(x)
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)


class DatasetIEMOCAP(Dataset):
    def __init__(self):
        self.t = normalize(np.load("features/text_bert.npy"))
        self.a = normalize(np.load("features/audio_wav2vec.npy"))
        self.v = normalize(np.load("features/video_processed.npy"))
        self.y = np.load("features/labels.npy")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.t[i], dtype=torch.float32),
            torch.tensor(self.a[i], dtype=torch.float32),
            torch.tensor(self.v[i], dtype=torch.float32),
            torch.tensor(self.y[i], dtype=torch.long)
        )


dataset = DatasetIEMOCAP()
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)


# ================= MODEL =================
class Model(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

        self.text_fc = nn.Sequential(
            nn.Linear(1536, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        self.audio_fc = nn.Sequential(
            nn.Linear(40, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        self.video_fc = nn.Sequential(
            nn.Linear(2304, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        # 🔥 GATED FUSION
        self.gate = nn.Sequential(
            nn.Linear(768 * 3, 3),
            nn.Softmax(dim=1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 4)
        )

    def forward(self, t, a, v):

        t = self.text_fc(t)
        a = self.audio_fc(a)
        v = self.video_fc(v)

        feats = torch.stack([t, a, v], dim=1)

        g = self.gate(torch.cat([t, a, v], dim=1)).unsqueeze(-1)

        x = (feats * g).sum(dim=1)

        return self.classifier(x)


# ================= TRAIN =================
def train_model(mode):

    print(f"\n🔥 Training {mode}")

    model = Model(mode).to(DEVICE)

    optimizer = Lion(model.parameters(), lr=LR)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    best = 0
    patience = 6
    no_imp = 0

    for epoch in range(EPOCHS):

        model.train()
        correct = total = 0

        for t, a, v, y in train_loader:
            t, a, v, y = t.to(DEVICE), a.to(DEVICE), v.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            out = model(t, a, v)

            loss = loss_fn(out, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        train_acc = correct / total * 100

        # VALID
        model.eval()
        correct = total = 0

        with torch.no_grad():
            for t, a, v, y in val_loader:
                t, a, v, y = t.to(DEVICE), a.to(DEVICE), v.to(DEVICE), y.to(DEVICE)

                out = model(t, a, v)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)

        val_acc = correct / total * 100

        scheduler.step()

        print(f"Epoch {epoch+1}: Train {train_acc:.1f}% | Val {val_acc:.1f}%")

        if val_acc > best:
            best = val_acc
            no_imp = 0
            torch.save(model.state_dict(), f"best_model_{mode}.pt")
        else:
            no_imp += 1

        if no_imp >= patience:
            print("🛑 Early stopping")
            break

    print(f"✅ Best ({mode}): {best:.2f}%")
    return best


# ================= RUN =================
modes = ["t", "a", "v", "ta", "tv", "av", "tav"]

results = {}

for m in modes:
    acc = train_model(m)
    results[m] = acc

# ===== FINAL SUMMARY =====
print("\n🔥 FINAL RESULTS 🔥")
for k, v in results.items():
    print(f"{k} → {v:.2f}%")
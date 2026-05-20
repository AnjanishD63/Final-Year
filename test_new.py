import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_PATH = 'best_model_tav.pt'
BATCH_SIZE = 4

print("Device:", DEVICE)


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


loader = DataLoader(DatasetIEMOCAP(), batch_size=BATCH_SIZE)


# ================= MODEL =================
class Model(nn.Module):
    def __init__(self):
        super().__init__()

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

        # ===== GATED FUSION =====
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


# ================= LOAD MODEL =================
print("Loading model...")

model = Model().to(DEVICE)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.eval()

print("✅ Model loaded")


# ================= TEST =================
preds = []
labels = []

with torch.no_grad():

    for t, a, v, y in loader:

        t = t.to(DEVICE)
        a = a.to(DEVICE)
        v = v.to(DEVICE)

        out = model(t, a, v)

        p = out.argmax(1)

        preds.extend(p.cpu().numpy())
        labels.extend(y.numpy())


# ================= RESULTS =================
acc = np.mean(np.array(preds) == np.array(labels)) * 100

print(f"\n🔥 FINAL TEST ACCURACY: {acc:.2f}%")

print("\n📊 Classification Report:\n")

print(classification_report(labels, preds))
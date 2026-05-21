"""Multimodal fusion model (text + audio MFCC + face-resized grayscale video patch)."""

import torch
import torch.nn as nn


class FusionEmotionModel(nn.Module):
    """Gated fusion of three modalities → 4-class logits (Angry, Happy, Sad, Neutral)."""

    def __init__(self) -> None:
        super().__init__()

        self.text_fc = nn.Sequential(
            nn.Linear(1536, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

        self.audio_fc = nn.Sequential(
            nn.Linear(40, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

        self.video_fc = nn.Sequential(
            nn.Linear(2304, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

        self.gate = nn.Sequential(nn.Linear(768 * 3, 3), nn.Softmax(dim=1))

        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 4),
        )

    def forward(self, t: torch.Tensor, a: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        t = self.text_fc(t)
        a = self.audio_fc(a)
        v = self.video_fc(v)

        feats = torch.stack([t, a, v], dim=1)
        g = self.gate(torch.cat([t, a, v], dim=1)).unsqueeze(-1)
        x = (feats * g).sum(dim=1)
        return self.classifier(x)

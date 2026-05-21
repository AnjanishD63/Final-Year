"""Legacy entry-point — forwards to the multimodal fused training loop."""

from pathlib import Path

from train_runner import train_fusion

if __name__ == "__main__":
    train_fusion(Path(__file__).resolve().parent)

"""Legacy evaluation shim — consumes best_model_tav.pt on the corpus under ./features."""

from pathlib import Path

from train_runner import evaluate_fusion

if __name__ == "__main__":
    evaluate_fusion(Path(__file__).resolve().parent)

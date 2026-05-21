#!/usr/bin/env python3
"""One-command launcher: installs deps → builds demo corpus → trains (if missing) → evaluates → launches Streamlit."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _install_deps() -> None:
    req_file = PROJECT_ROOT / "requirements.txt"

    print("[run_project] Installing python dependencies …")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "--requirement", str(req_file)], check=True)


def _launch_streamlit() -> None:
    app_path = PROJECT_ROOT / "app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--browser.gatherUsageStats=false",
    ]
    print("\n[run_project] Starting Streamlit web UI (`Ctrl+C` to stop) …")
    os.chdir(PROJECT_ROOT)
    subprocess.run(cmd, check=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bootstrap + train + demo UI for multimodal emotion project.")
    parser.add_argument("--skip-pip", action="store_true", help="Skip `pip install -r requirements.txt`.")
    parser.add_argument(
        "--fresh-data",
        action="store_true",
        help="Regenerate synthetic multimodal corpus under ./features even if markers exist.",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Erase best_model_tav.pt and train again.",
    )
    parser.add_argument(
        "--launch-ui-only",
        action="store_true",
        help="Only start Streamlit (expects features + checkpoints already present).",
    )
    opts = parser.parse_args(argv)

    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))

    if not opts.skip_pip:
        try:
            _install_deps()
        except subprocess.CalledProcessError:
            print(
                "[run_project] Pip install failed. Fix the error above or retry with `--skip-pip` after installing deps manually.",
                file=sys.stderr,
            )
            return 1

    if opts.launch_ui_only:
        _launch_streamlit()
        return 0

    from demo_features import build_demo_feature_pack
    from train_runner import evaluate_fusion, train_fusion

    build_demo_feature_pack(PROJECT_ROOT, regenerate=opts.fresh_data)

    ckpt = PROJECT_ROOT / "best_model_tav.pt"

    if opts.retrain and ckpt.exists():
        ckpt.unlink()
        print(f"[run_project] Removed stale checkpoint ({ckpt.name}).")

    if not ckpt.exists():
        print("\n[run_project] No fused checkpoint detected — kicking off training.")
        train_fusion(PROJECT_ROOT)
    else:
        print(f"[run_project] Found existing checkpoint `{ckpt.name}` — skipping training (use `--retrain` to override).")

    print("\n[run_project] Running quick corpus evaluation:")
    evaluate_fusion(PROJECT_ROOT)

    _launch_streamlit()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Desktop multimodal inference using OpenCV webcam + fused PyTorch checkpoint."""

from pathlib import Path

import cv2
import librosa
from model_def import FusionEmotionModel
from preprocessing import tensors_for_model, video_bgr_to_feature
from preprocessing import load_normalization_arrays
from transformers import BertModel, BertTokenizer


if not hasattr(cv2, "imshow") or "GUI:                           NONE" in cv2.getBuildInformation():
    raise RuntimeError(
        "OpenCV lacks GUI backends. Install `opencv-python` (not `opencv-python-headless`)."
    )

PROJECT_ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMOTIONS = ["Angry", "Happy", "Sad", "Neutral"]

STATS = load_normalization_arrays(PROJECT_ROOT)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)
bert.eval()

state_path = PROJECT_ROOT / "best_model_tav.pt"
if not state_path.exists():
    raise FileNotFoundError(f"{state_path} missing — run `python run_project.py` once to synthesize/train.")

model = FusionEmotionModel().to(DEVICE)
model.load_state_dict(torch.load(state_path, map_location=DEVICE))
model.eval()

current_emotion = "Waiting..."
cap = cv2.VideoCapture(0)

print("\nPress 'e' to detect emotion (requires sentence + WAV path) — 'q' to quit")


def load_audio_features(path: str) -> np.ndarray:
    samples, sr = librosa.load(path, sr=16000, mono=True)
    mfcc = librosa.feature.mfcc(y=samples.astype(np.float32), sr=int(sr), n_mfcc=40).mean(axis=1)
    return mfcc.astype(np.float32)


while True:
    ok, frame = cap.read()

    if not ok:
        break

    overlay = frame.copy()
    cv2.putText(overlay, f"Emotion: {current_emotion}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(
        overlay,
        "Press E = Detect | Q = Quit",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Emotion Detection", overlay)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("e"):
        cv2.destroyWindow("Emotion Detection")

        sentence = input("Sentence describing emotion: ").strip()
        wav_path = input("Path to audio clip (wav/mp3): ").strip()

        mfcc_vec = load_audio_features(wav_path)

        vid_vec = video_bgr_to_feature(frame, face_ok=True)

        t_tensor, a_tensor, v_tensor = tensors_for_model(
            sentence if sentence else "Neutral statement.",
            mfcc_vec,
            vid_vec,
            DEVICE,
            STATS,
            tokenizer,
            bert,
        )

        with torch.no_grad():
            logits = model(t_tensor, a_tensor, v_tensor)

        cls = logits.argmax(dim=1).item()
        current_emotion = EMOTIONS[cls]
        print(f"Predicted emotion: {current_emotion}")

        cv2.namedWindow("Emotion Detection")

    elif key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

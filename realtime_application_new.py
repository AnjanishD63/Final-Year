import cv2
import torch
import torch.nn as nn
import numpy as np
import librosa

from transformers import BertTokenizer

if not hasattr(cv2, 'imshow') or 'GUI:                           NONE' in cv2.getBuildInformation():
    raise RuntimeError(
        "OpenCV is installed without GUI support. "
        "Uninstall opencv-python-headless and install opencv-python so cv2.imshow() works."
    )

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_PATH = 'best_model_tav.pt'

EMOTIONS = ["Angry", "Happy", "Sad", "Neutral"]

FRAME_SIZE = 48
MAX_FRAMES = 30


# ================= TOKENIZER =================
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# ================= VIDEO =================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


def extract_video_frames(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        face = gray[y:y+h, x:x+w]
    else:
        face = gray

    face = cv2.resize(face, (FRAME_SIZE, FRAME_SIZE))

    face = face.astype(np.float32) / 255.0

    flat = face.flatten()

    return np.tile(flat, (MAX_FRAMES, 1))


# ================= AUDIO =================
def load_audio(path):

    audio, sr = librosa.load(path, sr=16000)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    mfcc = np.mean(mfcc, axis=1)

    return torch.tensor(
        mfcc,
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)


# ================= TEXT =================
def process_text(text):

    enc = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=50,
        return_tensors="pt"
    )

    return (
        enc['input_ids'].to(DEVICE),
        enc['attention_mask'].to(DEVICE)
    )


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

        g = self.gate(
            torch.cat([t, a, v], dim=1)
        ).unsqueeze(-1)

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

# ================= EMOTION DISPLAY =================
current_emotion = "Waiting..."


# ================= WEBCAM =================
cap = cv2.VideoCapture(0)

print("\n🎥 Press 'e' to detect emotion | 'q' to quit\n")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # ===== SHOW EMOTION ON SCREEN =====
    cv2.putText(
        frame,
        f"Emotion: {current_emotion}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Press E = Detect | Q = Quit",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("Emotion Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    # ===== DETECT =====
    if key == ord('e'):

        print("\n🔍 Detecting emotion...")

        # Pause camera window
        cv2.destroyWindow("Emotion Detection")

        # ===== TAKE INPUT EVERY TIME =====
        TEXT_INPUT = input("\nEnter sentence: ")

        AUDIO_PATH = input("Enter audio wav path: ")

        # ===== AUDIO =====
        a = load_audio(AUDIO_PATH)

        # ===== TEXT =====
        input_ids, attention_mask = process_text(TEXT_INPUT)

        # Placeholder text embedding
        t = torch.randn(1, 1536).to(DEVICE)

        # ===== VIDEO =====
        video_feat = extract_video_frames(frame)

        v = torch.tensor(
            video_feat,
            dtype=torch.float32
        ).mean(dim=0).unsqueeze(0).to(DEVICE)

        # ===== PREDICT =====
        with torch.no_grad():

            output = model(t, a, v)

            pred = torch.argmax(output, dim=1).item()

        current_emotion = EMOTIONS[pred]

        print(f"\n🎯 Emotion: {current_emotion}")

        # Reopen window
        cv2.namedWindow("Emotion Detection")

    # ===== QUIT =====
    elif key == ord('q'):
        break


cap.release()

cv2.destroyAllWindows()
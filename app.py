from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch

from model_def import FusionEmotionModel
from preprocessing import (
    EMOTION_LABELS,
    audio_bytes_to_feature,
    tensors_for_model,
    video_bgr_to_feature,
)
from preprocessing import load_normalization_arrays
from transformers import BertModel, BertTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="Multimodal Emotion Recognition", page_icon="🧠", layout="wide")


@st.cache_resource
def cached_normalization_stats():
    return load_normalization_arrays(PROJECT_ROOT)


@st.cache_resource
def cached_tokenizer_and_bert():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)
    bert.eval()
    return tokenizer, bert


@st.cache_resource
def cached_model():
    ckpt = PROJECT_ROOT / "best_model_tav.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"{ckpt} missing. Run `python run_project.py` once to synthesize data + train + launch."
        )

    model = FusionEmotionModel().to(DEVICE)
    sd = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(sd)
    model.eval()
    return model


def predict_multimodal(*, sentence: str, audio_bytes: bytes, frame_bgr: np.ndarray) -> tuple[str, float, np.ndarray]:
    stats_tup = cached_normalization_stats()
    tokenizer, bert = cached_tokenizer_and_bert()
    audio_vec = audio_bytes_to_feature(audio_bytes)
    vid_vec = video_bgr_to_feature(frame_bgr, face_ok=True)

    t, a, v = tensors_for_model(sentence.strip(), audio_vec, vid_vec, DEVICE, stats_tup, tokenizer, bert)

    model = cached_model()
    with torch.no_grad():
        logits = model(t, a, v)[0]

    probs = torch.softmax(logits, dim=-1).cpu().numpy()

    cls = int(probs.argmax())
    conf = float(probs[cls])

    label = EMOTION_LABELS[cls]
    return label, conf, probs


st.title("🧠 Multimodal Emotion Recognition")
st.markdown(
    "**Angry • Happy • Sad • Neutral** — fused text + spectro-temporal cues + webcam appearance. "
    "All three modalities are jointly scored (not independent single-modal models)."
)


with st.sidebar:
    st.markdown("### Runbook quick links")
    st.code(
        "# One-shot bootstrap:\npython run_project.py\n",
        language="bash",
    )
    if st.button("Refresh backend caches"):
        st.cache_resource.clear()
        st.success("Caches cleared.")


sentence = st.text_area("What are you expressing (text)?")
uploaded_audio = st.file_uploader("Upload WAV/MP3 for prosody cues", type=["wav", "mp3"])

st.subheader("Webcam snapshot (appearance cue)")
snapshot = st.camera_input("Snap a selfie / frame")


if st.button("Run fused multimodal inference", type="primary"):
    if not sentence.strip():
        st.error("Provide some text describing your emotional tone.")
        st.stop()
    if uploaded_audio is None:
        st.error("Upload audio so acoustic features can fuse with vision + language.")
        st.stop()
    if snapshot is None:
        st.error("Capture a webcam image for the spatial component.")
        st.stop()

    try:
        bytes_data = snapshot.getvalue()
        frame_bgr = cv2.imdecode(np.frombuffer(bytes_data, dtype=np.uint8), cv2.IMREAD_COLOR)

        payload = uploaded_audio.read()
        if hasattr(uploaded_audio, "seek"):
            uploaded_audio.seek(0)

        with st.spinner("Forward pass …"):
            label, confidence, probs = predict_multimodal(
                sentence=sentence,
                audio_bytes=payload,
                frame_bgr=frame_bgr,
            )

        st.success(f"Predicted **`{label}`** with softmax confidence **{confidence*100:.1f}%**")

        stacked = dict(zip(EMOTION_LABELS, probs.astype(float)))

        cols = st.columns(4)

        for col, emotion in zip(cols, EMOTION_LABELS):
            col.metric(label=emotion, value=f"{stacked[emotion]*100:.1f}%")

    except FileNotFoundError as exc:

        st.error(str(exc))


st.write("---")
st.caption("Tip: rerun `python run_project.py --retrain --fresh-data` if you regenerate the synthetic corpus.")

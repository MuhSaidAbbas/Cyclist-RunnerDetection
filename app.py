import streamlit as st
import cv2
import os
from ultralytics import YOLO

st.set_page_config(page_title="YOLO Sample Tracking", layout="centered")
st.title("YOLO Tracking – Video Sampel")

# =====================
# Load model
# =====================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# =====================
# Pilih video sampel
# =====================
SAMPLE_DIR = "samples"
videos = {
    "Video Sampel 1": os.path.join(SAMPLE_DIR, "Video3.mp4"),
    "Video Sampel 2": os.path.join(SAMPLE_DIR, "Video4.mp4"),
}

selected_video = st.selectbox(
    "Pilih video untuk diprediksi",
    list(videos.keys())
)

video_path = videos[selected_video]

st.video(video_path)

# =====================
# Predict
# =====================
if st.button("▶️ Mulai Tracking"):
    st.info("Processing video (short & controlled)...")

    cap = cv2.VideoCapture(video_path)
    frame_placeholder = st.empty()

    frame_count = 0
    MAX_FRAMES = 120  # batasi frame (AMAN)

    while cap.isOpened() and frame_count < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO inference (FRAME)
        results = model.predict(
            frame,
            imgsz=320,
            conf=0.5,
            verbose=False
        )

        annotated = results[0].plot()

        frame_placeholder.image(
            annotated,
            channels="BGR",
            caption=f"Frame {frame_count}"
        )

        frame_count += 1

    cap.release()
    st.success("✅ Tracking selesai (preview frame)")

import streamlit as st
import tempfile
import cv2
import os
from ultralytics import YOLO

st.set_page_config(page_title="YOLO Counter App", layout="centered")
st.title("Prediksi & Penghitungan Pelari dan Pesepeda")

# =====================
# Load model
# =====================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# =====================
# Upload video
# =====================
uploaded_file = st.file_uploader(
    "Upload video (mp4 / avi)",
    type=["mp4", "avi"]
)

if uploaded_file:
    st.success("Video berhasil di-upload")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    st.video(video_path)

    if st.button("▶️ Predict & Counting"):
        st.info("Processing video (frame sampling, CPU safe)...")

        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_interval = fps * 2  # 1 frame tiap 2 detik

        frame_idx = 0
        pelari_counts = []
        pesepeda_counts = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                results = model.predict(
                    frame,
                    imgsz=192,
                    conf=0.5,
                    verbose=False
                )

                if results and results[0].boxes is not None:
                    classes = results[0].boxes.cls.tolist()

                    pelari = classes.count(0)      # asumsi class 0 = Pelari
                    pesepeda = classes.count(1)    # asumsi class 1 = Pesepeda

                    pelari_counts.append(pelari)
                    pesepeda_counts.append(pesepeda)

            frame_idx += 1

        cap.release()

        # =====================
        # Final counting (stabil)
        # =====================
        total_pelari = max(pelari_counts) if pelari_counts else 0
        total_pesepeda = max(pesepeda_counts) if pesepeda_counts else 0

        st.success("✅ Prediksi selesai")
        st.metric("🏃 Total Pelari", total_pelari)
        st.metric("🚴 Total Pesepeda", total_pesepeda)

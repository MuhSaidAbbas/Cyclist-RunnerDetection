import streamlit as st
import cv2
import time
import tempfile
from ultralytics import YOLO
import subprocess
import os

# Memasangkan package FFMPEG untuk kompatibilitas video di Streamlit
FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe"  # path lokal pada laptop saya

def fix_video_for_streamlit(input_path, output_path):
    cmd = [
        FFMPEG_PATH,
        "-y",
        "-i", input_path,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "faststart",
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Konfigurasi model
MODEL_PATH = r"C:\DataOld\Kuliah\Semester7\ComputerVision\UASCV\runs\detect\yolo-model\weights\best.pt"

st.set_page_config(page_title="YOLO Counter App", layout="centered")
st.title("Prediksi dan Penghitungan Pelari & Pesepeda")

# Memuat model 
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# Mengunggah video
uploaded_file = st.file_uploader(
    "Upload video (mp4 / avi)",
    type=["mp4", "avi"]
)

if uploaded_file is not None:
    st.success("Video berhasil di-upload")

    # Menyimpan video sementara
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    input_video_path = tfile.name

    output_video_path = input_video_path.replace(".mp4", "_output.mp4")


    # Tombol untuk prediksi dan counting

    if st.button("▶️ Predict and Counting"):
        st.info("Processing video...")

        cap = cv2.VideoCapture(input_video_path)

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_video = cap.get(cv2.CAP_PROP_FPS)
        if fps_video == 0:
            fps_video = 25

        out = cv2.VideoWriter(
            output_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps_video,
            (w, h)
        )

    
        # Menghitung garis tengah 
    
        LINE_Y = int(h * 0.55)
        counted_ids = set()
        pelari_count = 0
        pesepeda_count = 0

        progress = st.progress(0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0

    
        # Proses frame demi frame
    
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            start = time.time()

            results = model.track(
                frame,
                conf=0.5,
                device=0,
                persist=True,
                verbose=False
            )

            for r in results:
                if r.boxes is None or r.boxes.id is None:
                    continue

                for box, track_id in zip(r.boxes, r.boxes.id):
                    track_id = int(track_id)
                    cls_id = int(box.cls[0])

                    if cls_id not in [0, 1]:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2   # Menyesuaikan dengan counter.py Saya

                    # Alur kontrol penghitungan
                    if LINE_Y - 15 <= cy <= LINE_Y + 15:
                        if track_id not in counted_ids:
                            counted_ids.add(track_id)
                            if cls_id == 0:
                                pelari_count += 1
                            else:
                                pesepeda_count += 1

                    # DRAW BOX
                    if cls_id == 0:
                        color = (0, 0, 255)       # merah (Pelari)
                        label = "Pelari"
                    else:
                        color = (0, 165, 255)     # oranye (Pesepeda)
                        label = "Pesepeda"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # INFO TEXT
            fps = 1 / (time.time() - start)
            cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Pelari: {pelari_count}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"Pesepeda: {pesepeda_count}", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            out.write(frame)

            current_frame += 1
            progress.progress(min(current_frame / frame_count, 1.0))

        cap.release()
        out.release()

    
        # Video fix untuk streamlit
    
        fixed_video_path = output_video_path.replace(".mp4", "_fixed.mp4")
        fix_video_for_streamlit(output_video_path, fixed_video_path)

        st.success("✅ Selesai diproses")

    
        # Menampilkan video hasil dan tombol dwonload
    
        st.video(fixed_video_path)

        with open(fixed_video_path, "rb") as f:
            st.download_button(
                "⬇️ Download hasil video",
                f,
                file_name="hasil_counting.mp4",
                mime="video/mp4"
            )

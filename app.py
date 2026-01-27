import streamlit as st
import tempfile
import os
from ultralytics import YOLO

st.set_page_config(page_title="YOLO Counter App", layout="centered")
st.title("Prediksi & Penghitungan Pelari dan Pesepeda")

# =====================
# Load model (RELATIVE PATH)
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

    # Simpan video ke temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    st.video(video_path)

    if st.button("▶️ Predict & Counting"):
        st.info("Processing video... (CPU mode)")

        # =====================
        # YOLO inference + tracking
        # =====================
        results = model.predict(
            source=video_path,
            conf=0.5,
            imgsz=320,
            vid_stride=2
        )

        # =====================
        # Ambil output video
        # =====================
        save_dir = results[0].save_dir
        output_video = None

        for f in os.listdir(save_dir):
            if f.endswith(".mp4"):
                output_video = os.path.join(save_dir, f)
                break

        if output_video:
            st.success("✅ Selesai diproses")
            st.video(output_video)

            with open(output_video, "rb") as f:
                st.download_button(
                    "⬇️ Download hasil video",
                    f,
                    file_name="hasil_counting.mp4",
                    mime="video/mp4"
                )
        else:
            st.error("Output video tidak ditemukan.")





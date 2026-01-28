import streamlit as st
import os
from ultralytics import YOLO

st.set_page_config(page_title="YOLO Sample Tracking", layout="centered")
st.title("YOLO Tracking – Video Sampel")

st.expander("ℹ️ Informasi Keterbatasan Sistem & Metode Prediksi", expanded=False).markdown(
    """
**Catatan Penting Mengenai Hasil Prediksi**

Aplikasi ini dijalankan menggunakan **lingkungan komputasi CPU gratis (tanpa GPU)** pada platform cloud. 
Kondisi ini menimbulkan beberapa keterbatasan teknis yang perlu dipahami:

1. **Inferensi Berbasis CPU**
   - Model YOLO pada dasarnya dioptimalkan untuk GPU.
   - Pada CPU, proses inferensi video berskala besar menjadi jauh lebih lambat dan berisiko menyebabkan kegagalan sistem (timeout atau memory overload).

2. **Pembatasan Prediksi Video**
   - Untuk menjaga stabilitas aplikasi, prediksi tidak dilakukan pada seluruh frame video secara berurutan.
   - Sistem menggunakan **frame sampling** (pengambilan frame secara berkala), bukan pemrosesan video penuh.

3. **Bounding Box dan Tracking ID**
   - Bounding box dihasilkan pada frame-frame tertentu, namun tidak ditampilkan sebagai video output penuh.
   - **Tracking ID (identitas objek)** tidak diaktifkan karena memerlukan algoritma tracking tambahan (misalnya ByteTrack) yang tidak stabil pada lingkungan cloud gratis.

4. **Hasil Counting**
   - Nilai penghitungan objek bersifat **estimasi kepadatan (instantaneous count)**, bukan jumlah unik individu sepanjang video.
   - Pendekatan ini dipilih agar hasil tetap informatif tanpa mengorbankan stabilitas sistem.

5. **Tujuan Implementasi**
   - Aplikasi ini difokuskan sebagai **demonstrasi konsep dan metode**, bukan sistem produksi berskala besar.
   - Untuk performa optimal (bounding box penuh, tracking ID stabil, dan video output lengkap), disarankan menjalankan sistem pada **lingkungan lokal atau server GPU**.

Pendekatan ini merupakan praktik umum dalam deployment sistem computer vision berbasis cloud dengan sumber daya terbatas.
"""
)

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


st.expander("🎥 Contoh Output Ideal (Server / Lokal dengan Resource Lebih Baik)", expanded=False).markdown(
    """
Video berikut merupakan **contoh hasil prediksi ideal** yang diperoleh ketika model dijalankan
pada **lingkungan lokal atau server dengan resource komputasi lebih baik (GPU / CPU kuat)**.

Pada kondisi tersebut, sistem mampu menghasilkan:
- Bounding box yang stabil
- Tracking ID untuk setiap objek
- Counting berbasis lintasan objek
- Output video lengkap per frame

Video ini **ditampilkan sebagai referensi visual**, bukan hasil prediksi langsung dari sistem cloud ini.
"""
)

# Path ke contoh output ideal
IDEAL_OUTPUT_VIDEO = os.path.join("samples", "Output Video.mp4")

if os.path.exists(IDEAL_OUTPUT_VIDEO):
    st.video(IDEAL_OUTPUT_VIDEO)
else:
    st.warning("Video contoh output ideal tidak ditemukan.")





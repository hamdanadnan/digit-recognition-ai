import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Digit Recognition AI",
    page_icon="✍️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_model.h5")

model = load_model()

# ---------------- SOFT DARK THEME CSS ----------------
st.markdown("""
<style>
body {
    background-color: #0f172a;
    color: #f8fafc;
}
.main {
    background-color: #0f172a;
}
h1, h2, h3, h4 {
    color: #f8fafc;
}
.card {
    background: #111827;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #1f2933;
    margin-bottom: 20px;
}
.info {
    background: #0b1220;
    border-left: 5px solid #38bdf8;
    padding: 12px;
    border-radius: 8px;
    color: #e5e7eb;
}
.result {
    background: #052e1a;
    border-left: 5px solid #22c55e;
    padding: 14px;
    border-radius: 10px;
    font-size: 20px;
    font-weight: bold;
    color: #dcfce7;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("<h1 style='text-align:center;'>✍️ Digit Recognition AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#9ca3af;'>Draw or upload a handwritten digit (0–9)</p>", unsafe_allow_html=True)

# ---------------- INSTRUCTIONS ----------------
st.markdown("""
<div class="card info">
<b>Instructions</b><br>
• Draw <b>WHITE digit</b> on <b>BLACK background</b><br>
• Draw in the <b>center</b><br>
• Use <b>thick strokes</b><br>
• Draw only <b>one digit</b><br>
• Uploaded image must also be <b>white on black</b>
</div>
""", unsafe_allow_html=True)

# ---------------- TABS ----------------
tab1, tab2 = st.tabs(["✏️ Draw Digit", "📤 Upload Image"])

# ================= DRAW TAB =================
with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=18,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        img = canvas_result.image_data.astype(np.uint8)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (28, 28))
        gray = gray / 255.0
        gray = gray.reshape(1, 28, 28, 1)

        prediction = model.predict(gray)
        digit = int(np.argmax(prediction))

        st.markdown(
            f"<div class='result'>🧠 Predicted Digit: {digit}</div>",
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ================= UPLOAD TAB =================
with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload an image (PNG / JPG)",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("L")
        img = np.array(img)
        img = cv2.resize(img, (28, 28))
        img = img / 255.0
        img = img.reshape(1, 28, 28, 1)

        prediction = model.predict(img)
        digit = int(np.argmax(prediction))

        st.image(uploaded_file, caption="Uploaded Image", width=150)
        st.markdown(
            f"<div class='result'>🧠 Predicted Digit: {digit}</div>",
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown(
    "<p style='text-align:center; color:#64748b;'>Built with Streamlit & TensorFlow</p>",
    unsafe_allow_html=True
)

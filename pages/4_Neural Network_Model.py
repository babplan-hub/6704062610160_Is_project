import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

st.set_page_config(page_title="AI Dog Breed Classifier", layout="wide")

# =========================
# AI DARK UI STYLE
# =========================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
}

.big-title {
    font-size: 46px;
    font-weight: 800;
    background: -webkit-linear-gradient(#00F5A0,#00D9F5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    font-size:18px;
    color:#cfd8dc;
}

.card {
    background: rgba(255,255,255,0.06);
    padding: 30px;
    border-radius: 20px;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.15);
    margin-bottom:25px;
}

.result-box {
    text-align:center;
    padding:30px;
    border-radius:20px;
    background: rgba(0,255,200,0.08);
    border:1px solid rgba(0,255,200,0.3);
}

.breed-name {
    font-size:32px;
    font-weight:700;
    color:#00F5A0;
}

.confidence {
    font-size:20px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/dog_model.h5")

model = load_model()

with open("model/class_names.json", "r") as f:
    class_names = json.load(f)

# =========================
# HEADER
# =========================
st.markdown('<div class="big-title">🐶 AI Dog Breed Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Deep Learning Image Recognition System</div>', unsafe_allow_html=True)
st.write("")

# =========================
# UPLOAD SECTION
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("📤 Upload Dog Image", type=["jpg","png","jpeg"])
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# PREDICTION
# =========================
if uploaded_file:

    img = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1,1])

    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)

    with col2:
        with st.spinner("🧠 AI analyzing image..."):

            img_resized = img.resize((224,224))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            preds = model.predict(img_array)
            idx = np.argmax(preds)
            confidence = preds[0][idx] * 100
            breed_name = class_names[idx]

        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown(f'<div class="breed-name">{breed_name}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="confidence">Confidence: {confidence:.2f}%</div>', unsafe_allow_html=True)
        st.progress(float(confidence/100))
        st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # TOP 3 PREDICTIONS
    # =========================
    st.write("")
    st.subheader("📊 Top 3 Predictions")

    top3_idx = np.argsort(preds[0])[-3:][::-1]

    for i in top3_idx:
        st.write(f"{class_names[i]} — {preds[0][i]*100:.2f}%")
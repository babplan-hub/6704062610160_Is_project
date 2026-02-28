import streamlit as st

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Intelligent AI System",
    layout="wide",
    page_icon="🤖"
)

# -------------------------
# CUSTOM CSS (AI STYLE)
# -------------------------
st.markdown("""
<style>

body {
    background-color: #0e1117;
}

.main {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    padding: 2rem;
    border-radius: 15px;
}

.big-title {
    font-size: 42px;
    font-weight: 700;
    background: -webkit-linear-gradient(#00F5A0, #00D9F5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    font-size:18px;
    color: #cfd8dc;
}

.card {
    background: rgba(255, 255, 255, 0.05);
    padding: 25px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    transition: 0.3s;
    border: 1px solid rgba(255,255,255,0.1);
}

.card:hover {
    transform: scale(1.03);
    border: 1px solid #00F5A0;
}

.card-title {
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# HEADER
# -------------------------
st.markdown('<div class="big-title">🚀 Intelligent AI Prediction Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Advanced Machine Learning & Neural Network System</div>', unsafe_allow_html=True)

st.write("")
st.write("")

# -------------------------
# CARDS LAYOUT
# -------------------------
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

with col1:
    st.markdown("""
    <div class="card">
        <div class="card-title">📊 ML Model Description</div>
        Overview of traditional Machine Learning algorithms used in this system.
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <div class="card-title">🧠 Neural Network Model Description</div>
        Deep Learning architecture explanation and training strategy.
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <div class="card-title">🧪 ML Model Test</div>
        Test real-time predictions using Machine Learning approach.
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="card">
        <div class="card-title">⚡ Neural Network Model Test</div>
        Evaluate image classification using deep neural network.
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.markdown(
    "<center style='color:gray'>Use the navigation menu on the left to explore the AI system.</center>",
    unsafe_allow_html=True
)
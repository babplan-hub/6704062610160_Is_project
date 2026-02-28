import streamlit as st

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="NBA AI Prediction Platform",
    layout="wide",
    page_icon="🏀"
)

# -------------------------
# CUSTOM AI CSS
# -------------------------
st.markdown("""
<style>

body {
    background-color: #0e1117;
}

.main {
    background: linear-gradient(135deg, #141e30, #243b55);
    padding: 2rem;
    border-radius: 20px;
}

.big-title {
    font-size: 44px;
    font-weight: 800;
    background: -webkit-linear-gradient(#ff9966, #ff5e62);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    font-size:18px;
    color: #cfd8dc;
}

.section-card {
    background: rgba(255,255,255,0.05);
    padding: 25px;
    border-radius: 18px;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 25px;
    transition: 0.3s;
}

.section-card:hover {
    transform: scale(1.01);
    border: 1px solid #ff9966;
}

.metric-box {
    background: rgba(255,255,255,0.07);
    padding: 20px;
    border-radius: 15px;
    text-align:center;
}

.metric-number {
    font-size:28px;
    font-weight:700;
    color:#ff9966;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# HEADER
# -------------------------
st.markdown('<div class="big-title">🏀 NBA Advanced Prediction Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Hybrid Machine Learning & ELO Rating System</div>', unsafe_allow_html=True)

st.write("")

# -------------------------
# METRICS ROW
# -------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-box">
        <div class="metric-number">65–72%</div>
        Model Accuracy
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-box">
        <div class="metric-number">Time-Series</div>
        Data Split Strategy
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-box">
        <div class="metric-number">No Leakage</div>
        Training Integrity
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

# -------------------------
# SECTION 1
# -------------------------
st.markdown("""
<div class="section-card">
<h3>📊 Hybrid NBA Prediction System</h3>

This platform predicts NBA game outcomes using:

<b>🔹 ELO Rating System</b><br>
Dynamic team strength rating updated sequentially.<br><br>

<b>🔹 Hybrid Logistic Regression Model</b><br>
Features used:
<ul>
<li>ELO Difference</li>
<li>FG% Difference</li>
<li>Rebound Difference</li>
<li>Assist Difference</li>
<li>FT% Difference</li>
<li>3PT% Difference</li>
</ul>

<b>🔒 Training Strategy</b><br>
• Time-series split (shuffle=False)<br>
• No data leakage<br>
• StandardScaler normalization
</div>
""", unsafe_allow_html=True)

# -------------------------
# SECTION 2
# -------------------------
st.markdown("""
<div class="section-card">
<h3>🧠 Development Methodology</h3>

<b>1️⃣ Data Preparation</b><br>
• Sorted by GAME_DATE_EST<br>
• Created HOME_TEAM_WINS target<br>
• Generated statistical differences<br>
• Removed missing values<br>
• Time-Series split (80/20)<br><br>

<b>2️⃣ ELO Rating System</b><br>
Expected Score Formula:<br>
<i>1 / (1 + 10^((Opponent - Team)/400))</i><br><br>
Home Advantage: +65 ELO<br><br>

<b>3️⃣ Logistic Regression Model</b><br>
Probability Function:<br>
<i>P(Y=1) = 1 / (1 + e^-(β0 + β1X1 + ... + βnXn))</i><br><br>

• Suitable for binary classification<br>
• Interpretable coefficients<br>
• Stable for structured sports data
</div>
""", unsafe_allow_html=True)

# -------------------------
# SECTION 3
# -------------------------
st.markdown("""
<div class="section-card">
<h3>📈 Model Evaluation</h3>

• Accuracy Metric<br>
• Time-Series Validation<br>
• No Data Leakage<br><br>

Model performance range: <b>65–72%</b><br>
(High realism for sports prediction problems)
</div>
""", unsafe_allow_html=True)

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.success("🔒 Current Production Model: Hybrid Logistic Model")
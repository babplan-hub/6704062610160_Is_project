import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="NBA Hybrid AI Predictor", layout="wide")

# =========================
# PRO DARK AI CSS
# =========================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
}

.main {
    background: rgba(0,0,0,0);
}

.big-title {
    font-size: 48px;
    font-weight: 800;
    background: -webkit-linear-gradient(#ff9966,#ff5e62);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.card {
    background: rgba(255,255,255,0.06);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.15);
    margin-bottom:20px;
}

.score-card {
    background: rgba(255,255,255,0.08);
    padding: 30px;
    border-radius: 20px;
    text-align: center;
}

.team-name {
    font-size: 22px;
    font-weight: bold;
    color: white;
}

.score-number {
    font-size: 64px;
    font-weight: 800;
    color: #ff9966;
}

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🏀 NBA Hybrid AI Prediction Engine</div>', unsafe_allow_html=True)
st.write("")

# =========================
# LOAD DATA
# =========================
@st.cache_resource
def load_all():
    games = pd.read_csv("games.csv")
    teams = pd.read_csv("teams.csv")
    model = joblib.load("model_class.pkl")
    scaler = joblib.load("scaler.pkl")
    return games, teams, model, scaler

games, teams, model, scaler = load_all()
games = games.sort_values("GAME_DATE_EST")

teams["FULL_NAME"] = teams["CITY"] + " " + teams["NICKNAME"]
team_names = sorted(teams["FULL_NAME"].unique())

col1, col2 = st.columns(2)

with col1:
    home_team = st.selectbox("🏠 Home Team", team_names)

with col2:
    away_team = st.selectbox("✈ Away Team", team_names)

if home_team == away_team:
    st.warning("Please select different teams.")

# =========================
# ELO
# =========================
K = 20
INITIAL_ELO = 1500
HOME_ADV = 65

def calculate_elo():
    elo = {}
    for _, row in games.iterrows():
        h = row["HOME_TEAM_ID"]
        a = row["VISITOR_TEAM_ID"]
        win = row["HOME_TEAM_WINS"]

        if h not in elo: elo[h] = INITIAL_ELO
        if a not in elo: elo[a] = INITIAL_ELO

        eh = 1/(1+10**(((elo[a])-(elo[h]+HOME_ADV))/400))
        elo[h] += K*(win-eh)
        elo[a] += K*((1-win)-(1-eh))
    return elo

elo_dict = calculate_elo()

# =========================
# PREDICTION
# =========================
if st.button("🔥 Run AI Prediction") and home_team != away_team:

    with st.spinner("AI analyzing team performance..."):
        
        home_row = teams[teams["FULL_NAME"] == home_team]
        away_row = teams[teams["FULL_NAME"] == away_team]

        home_id = home_row["TEAM_ID"].values[0]
        away_id = away_row["TEAM_ID"].values[0]

        home_logo = f"https://cdn.nba.com/logos/nba/{home_id}/global/L/logo.svg"
        away_logo = f"https://cdn.nba.com/logos/nba/{away_id}/global/L/logo.svg"

        elo_diff = elo_dict.get(home_id,1500) - elo_dict.get(away_id,1500)

        last_home = games[games["HOME_TEAM_ID"]==home_id].tail(1)
        last_away = games[games["VISITOR_TEAM_ID"]==away_id].tail(1)

        fg_diff = last_home["FG_PCT_home"].values[0] - last_away["FG_PCT_away"].values[0]
        reb_diff = last_home["REB_home"].values[0] - last_away["REB_away"].values[0]
        ast_diff = last_home["AST_home"].values[0] - last_away["AST_away"].values[0]
        ft_diff = last_home["FT_PCT_home"].values[0] - last_away["FT_PCT_away"].values[0]
        fg3_diff = last_home["FG3_PCT_home"].values[0] - last_away["FG3_PCT_away"].values[0]

        X = np.array([[elo_diff, fg_diff, reb_diff, ast_diff, ft_diff, fg3_diff]])
        X = scaler.transform(X)

        prob = model.predict_proba(X)[0][1]

    # =========================
    # WIN PROBABILITY CARD
    # =========================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 AI Win Probability")

    c1, c2 = st.columns(2)
    with c1:
        st.metric(home_team, f"{prob*100:.2f}%")
    with c2:
        st.metric(away_team, f"{(1-prob)*100:.2f}%")

    st.progress(float(prob))
    st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # SCORE PREDICTION
    # =========================
    base_score = 110
    score_diff = (prob - 0.5) * 24

    home_score = base_score + score_diff
    away_score = base_score - score_diff

    st.subheader("🏟 Predicted Final Score")

    col1, col2 = st.columns(2)

    with col1:
        st.image(home_logo, width=130)
        st.markdown(f"""
        <div class="score-card">
            <div class="team-name">{home_team}</div>
            <div class="score-number">{round(home_score)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.image(away_logo, width=130)
        st.markdown(f"""
        <div class="score-card">
            <div class="team-name">{away_team}</div>
            <div class="score-number">{round(away_score)}</div>
        </div>
        """, unsafe_allow_html=True)

    # =========================
    # QUARTER BREAKDOWN
    # =========================
    st.subheader("⏱ Estimated Quarter Breakdown")

    for q in range(1,5):
        st.write(f"Q{q}: {round(home_score/4)} - {round(away_score/4)}")
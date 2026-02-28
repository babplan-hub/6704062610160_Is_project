import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

print("Loading data...")

games = pd.read_csv("games.csv")
games = games.sort_values("GAME_DATE_EST")

# ===============================
# PARAMETERS
# ===============================
K = 20
INITIAL_ELO = 1500
HOME_ADV = 65

elo_dict = {}

elo_home_list = []
elo_away_list = []

# ===============================
# CALCULATE ELO SEQUENTIALLY
# ===============================
for _, row in games.iterrows():

    home = row["HOME_TEAM_ID"]
    away = row["VISITOR_TEAM_ID"]
    home_win = row["HOME_TEAM_WINS"]

    if home not in elo_dict:
        elo_dict[home] = INITIAL_ELO
    if away not in elo_dict:
        elo_dict[away] = INITIAL_ELO

    elo_home = elo_dict[home]
    elo_away = elo_dict[away]

    elo_home_list.append(elo_home)
    elo_away_list.append(elo_away)

    expected_home = 1 / (1 + 10 ** (((elo_away) - (elo_home + HOME_ADV)) / 400))

    elo_dict[home] = elo_home + K * (home_win - expected_home)
    elo_dict[away] = elo_away + K * ((1 - home_win) - (1 - expected_home))

games["elo_home"] = elo_home_list
games["elo_away"] = elo_away_list
games["elo_diff"] = games["elo_home"] - games["elo_away"]

# ===============================
# STAT DIFFERENCES
# ===============================
games["fg_diff"] = games["FG_PCT_home"] - games["FG_PCT_away"]
games["reb_diff"] = games["REB_home"] - games["REB_away"]
games["ast_diff"] = games["AST_home"] - games["AST_away"]
games["ft_diff"] = games["FT_PCT_home"] - games["FT_PCT_away"]
games["fg3_diff"] = games["FG3_PCT_home"] - games["FG3_PCT_away"]

# ===============================
# FEATURE SET
# ===============================
features = [
    "elo_diff",
    "fg_diff",
    "reb_diff",
    "ast_diff",
    "ft_diff",
    "fg3_diff"
]

games = games.dropna(subset=features)   # 🔥 แก้ NaN ตรงนี้

X = games[features]
y = games["HOME_TEAM_WINS"]

# ===============================
# TIME-SERIES SPLIT (NO SHUFFLE)
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ===============================
# SCALE FEATURES (ช่วย Logistic)
# ===============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================
# TRAIN MODEL
# ===============================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ===============================
# EVALUATION
# ===============================
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

print("======================================")
print(f"Hybrid Model Accuracy: {acc*100:.2f}%")
print("======================================")

# ===============================
# SAVE MODEL + SCALER
# ===============================
joblib.dump(model, "model_class.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Hybrid model + scaler saved successfully.")
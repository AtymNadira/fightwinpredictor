import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Fight Outcome Predictor", layout="centered")
st.title("🥊 Fight Outcome Predictor")

@st.cache_data
def load_data():
    # Load raw fight data once
    df = pd.read_csv("data.csv")
    return df

@st.cache_resource
def train_model(df: pd.DataFrame):
    # 1. Map Winner → numeric Outcome
    df = df.copy()
    df["Winner"] = df["Winner"].astype(str).str.strip().str.lower()
    mapping = {"red": 1, "blue": 2, "draw": 0, "no contest": 0, "nc": 0}
    df["Outcome"] = df["Winner"].map(mapping).fillna(0).astype(int)

    # 2. Encode stances
    le = LabelEncoder()
    all_stances = pd.concat([
        df["R_Stance"].astype(str).str.lower(),
        df["B_Stance"].astype(str).str.lower()
    ]).unique()
    le.fit(all_stances)
    df["R_Stance_enc"] = le.transform(df["R_Stance"].astype(str).str.lower())
    df["B_Stance_enc"] = le.transform(df["B_Stance"].astype(str).str.lower())

    # 3. Build difference features
    df["height_diff"] = df["R_Height_cms"] - df["B_Height_cms"]
    df["weight_diff"] = df["R_Weight_lbs"] - df["B_Weight_lbs"]
    df["reach_diff"]  = df["R_Reach_cms"]  - df["B_Reach_cms"]
    df["age_diff"]    = df["R_age"]        - df["B_age"]

    # 4. Assemble training matrix
    features = [
        "height_diff",
        "weight_diff",
        "reach_diff",
        "age_diff",
        "R_Stance_enc",
        "B_Stance_enc",
    ]
    X = df[features]
    y = df["Outcome"]

    # 5. Train RandomForest once
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, le

# Load and train
df = load_data()
model, le_stance = train_model(df) #Hello

# Sidebar inputs
st.sidebar.header("Enter Fighter Parameters")
st.sidebar.subheader("🔴 Red Corner")
R_height = st.sidebar.number_input("Height (cm)", 100.0, 250.0, 175.0)
R_weight = st.sidebar.number_input("Weight (lbs)", 100.0, 300.0, 170.0)
R_reach  = st.sidebar.number_input("Reach (cm)", 100.0, 250.0, 180.0)
R_age    = st.sidebar.number_input("Age", 18, 60, 30)
R_stance = st.sidebar.selectbox("Stance", [s.title() for s in le_stance.classes_])

st.sidebar.subheader("🔵 Blue Corner")
B_height = st.sidebar.number_input("Height (cm)", 100.0, 250.0, 180.0, key="B1")
B_weight = st.sidebar.number_input("Weight (lbs)", 100.0, 300.0, 175.0, key="B2")
B_reach  = st.sidebar.number_input("Reach (cm)", 100.0, 250.0, 185.0, key="B3")
B_age    = st.sidebar.number_input("Age", 18, 60, 28, key="B4")
B_stance = st.sidebar.selectbox("Stance", [s.title() for s in le_stance.classes_], key="B5")

if st.sidebar.button("Predict outcome"):
    # Encode stances
    R_enc = le_stance.transform([R_stance.lower()])[0]
    B_enc = le_stance.transform([B_stance.lower()])[0]

    # Compute diff features
    feats = np.array([[
        R_height - B_height,
        R_weight - B_weight,
        R_reach  - B_reach,
        R_age    - B_age,
        R_enc,
        B_enc
    ]])

    pred = model.predict(feats)[0]
    if pred == 1:
        result = "🔴 Red corner wins!"
    elif pred == 2:
        result = "🔵 Blue corner wins!"
    else:
        result = "🤝 It's a Draw / No Contest"

    st.markdown("## Prediction")
    st.success(result)

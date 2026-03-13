import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# ── Train models on cached data so they load once ────────────────────────────
@st.cache_resource
def train_models():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)

    df['Age'] = pd.to_numeric(df['Age'], errors='coerce').fillna(df['Age'].median())
    df['Fare'] = pd.to_numeric(df['Fare'], errors='coerce').fillna(0)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    X = df[features].fillna(0)
    y = df['Survived']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LogisticRegression(max_iter=200)
    lr.fit(X_train, y_train)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(X_train, y_train)

    return lr, rf, gb

lr, rf, gb = train_models()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🚢 Titanic Survival Predictor")
st.markdown("**Kaggle Competition Project** — 81% Validation Accuracy · 0.76076 Kaggle Score (Top 45%)")
st.divider()

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("🎫 Passenger Class", [1, 2, 3],
                          help="1 = First class, 2 = Second, 3 = Third")
    sex = st.selectbox("👤 Sex", ["Male", "Female"])
    age = st.slider("🎂 Age", 1, 80, 25)

with col2:
    fare = st.slider("💰 Fare Paid (£)", 0, 512, 32)
    sibsp = st.number_input("👫 Siblings / Spouses aboard", 0, 8, 0)
    parch = st.number_input("👨‍👧 Parents / Children aboard", 0, 6, 0)

model_choice = st.selectbox("🤖 Choose Model", 
    ["Logistic Regression (Best — 81.0%)", "Random Forest (80.4%)", "Gradient Boosting (80.4%)"])

st.divider()

if st.button("🔍 Predict Survival", use_container_width=True, type="primary"):
    sex_val = 0 if sex == "Male" else 1
    features = np.array([[pclass, sex_val, age, sibsp, parch, fare]])

    model_map = {
        "Logistic Regression (Best — 81.0%)": lr,
        "Random Forest (80.4%)": rf,
        "Gradient Boosting (80.4%)": gb
    }
    model = model_map[model_choice]
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    st.markdown("### Result")
    if prediction == 1:
        st.success(f"✅ **SURVIVED** — {probability*100:.1f}% survival probability")
        st.balloons()
    else:
        st.error(f"❌ **DID NOT SURVIVE** — {probability*100:.1f}% survival probability")

    st.progress(float(probability), text=f"Survival probability: {probability*100:.1f}%")

# ── Model comparison table ────────────────────────────────────────────────────
with st.expander("📊 Model Comparison"):
    st.table(pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "Gradient Boosting"],
        "Validation Accuracy": ["81.0% ⭐", "80.4%", "80.4%"],
        "Kaggle Score": ["0.76076", "—", "—"]
    }))

st.caption("Built by [Astha Chaudhary] · [GitHub](https://github.com/castha25) · Kaggle Titanic Competition")

import streamlit as st

st.set_page_config(
    page_title="Diabetes Risk Prediction Dashboard",
    page_icon="⚕️",
    layout="wide"
)

st.markdown(
    """
    <style>
    /* FORCE SMALLER FONTS - Override everything */
    * {
        font-size: 0.85rem !important;
    }
    h1 { font-size: 1.3rem !important; }
    h2 { font-size: 1.2rem !important; }
    h3 { font-size: 1.1rem !important; }
    
    /* Top bar - smaller name */
    #top-bar {
        font-size: 1.0rem !important;
        height: 35px !important;
    }
    
    /* Metrics - smaller numbers */
    .tall-metric-main {
        font-size: 1.6rem !important;
    }
    .tall-metric-label {
        font-size: 0.9rem !important;
    }
    
    /* Sidebar labels smaller */
    section[data-testid="stSidebar"] label {
        font-size: 0.95rem !important;
    }
    
    /* Keep your glass styling but smaller padding */
    .tall-metric {
        padding: 16px 14px !important;
        min-height: 180px !important;
    }
    .main-hero {
        padding: 16px 20px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div id='top-bar'>Diabetes Risk Dashboard</div>", unsafe_allow_html=True)

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

st.markdown("""
<style>
/* Your existing beautiful styling - just smaller */
.stApp {
    background: radial-gradient(circle at top left, #e0f2fe 0, #e5e7eb 45%, #f9fafb 100%);
    font-family: "Inter", sans-serif;
}
.main .block-container {
    padding-top: 50px;
}
.tall-metric {
    background: rgba(255,255,255,0.82);
    border-radius: 20px;
    box-shadow: 0 12px 30px rgba(15,23,42,0.2);
    border: 1px solid rgba(191,219,254,0.8);
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def train_model():
    np.random.seed(42)
    n = 5000
    data = pd.DataFrame({
        "HighBP": np.random.binomial(1, 0.3, n),
        "HighChol": np.random.binomial(1, 0.25, n),
        "BMI": np.clip(np.random.normal(28, 6, n), 15, 55),
        "Age": np.random.choice(range(1, 14), n),
        "GenHlth": np.random.choice(range(1, 6), n),
        "PhysHlth": np.random.randint(0, 30, n),
        "MentHlth": np.random.randint(0, 30, n),
        "PhysActivity": np.random.binomial(1, 0.65, n),
        "Smoker": np.random.binomial(1, 0.2, n),
    })
    y = (
        (data["BMI"] > 30) * 0.5
        + (data["HighBP"]) * 0.3
        + (data["GenHlth"] > 3) * 0.25
        + (data["Smoker"]) * 0.15
        + np.random.normal(0, 0.2, n)
    ).clip(0, 1)
    y = (np.random.random(n) < y).astype(int)

    scaler = StandardScaler()
    X = scaler.fit_transform(
        data[["HighBP","HighChol","BMI","Age","GenHlth","PhysHlth","MentHlth","PhysActivity","Smoker"]]
    )
    model = LogisticRegression(max_iter=500).fit(X, y)
    return model, scaler

model, scaler = train_model()

AGE_OPTS = {i: f"Age {18+((i-1)*5)}-{22+((i-1)*5)}" for i in range(1, 14)}
AGE_OPTS[13] = "Age 80+"

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=70)
    st.title("Patient Profile")
    st.caption("Fill in profile for personalized screening")

    with st.expander("👤 Demographics", expanded=True):
        age = st.selectbox("Age Group", list(AGE_OPTS.keys()), format_func=lambda x: AGE_OPTS[x], index=8)
        sex = st.radio("Sex", ["Female", "Male"], horizontal=True)
        income = st.select_slider("Income (1–8)", range(1, 9), 5)

    with st.expander("🩺 Clinical Metrics", expanded=True):
        bmi = st.number_input("BMI", 10.0, 60.0, 25.5, 0.1)
        gen_hlth = st.select_slider("General Health (1–5)", range(1, 6), 3)
        high_bp = st.checkbox("High Blood Pressure")
        high_chol = st.checkbox("High Cholesterol")
        phys_hlth = st.slider("Physical Health Days (0–30)", 0, 30, 0)

    with st.expander("🧠 Mental Health", expanded=False):
        ment_hlth = st.slider("Mental Health Days (0–30)", 0, 30, 0)

    with st.expander("🌱 Lifestyle & History", expanded=False):
        phys_act = st.checkbox("Physically Active (150+ min/week)", True)
        smoker = st.checkbox("Current Smoker")
        stroke = st.checkbox("Stroke History")
        heart = st.checkbox("Heart Disease")

st.success("✅ Advanced ML | Diabetes Risk Screening")

hero_left, hero_right = st.columns([2.2, 1])

with hero_left:
    st.markdown("""
    <div class="main-hero">
        <h1>⚕️ Diabetes Risk Dashboard</h1>
        <p>Intelligent risk screening using clinical + lifestyle data. Educational tool only.</p>
    </div>
    """, unsafe_allow_html=True)

with hero_right:
    st.metric("Focus", "Type 2 Diabetes")

input_data = {
    "HighBP": int(high_bp), "HighChol": int(high_chol), "BMI": bmi, "Age": age,
    "GenHlth": gen_hlth, "PhysHlth": phys_hlth, "MentHlth": ment_hlth,
    "PhysActivity": int(phys_act), "Smoker": int(smoker)
}

X = pd.DataFrame([input_data])
X_scaled = scaler.transform(X)
prob = model.predict_proba(X_scaled)[0, 1]
pred_class = model.predict(X_scaled)[0]

if prob < 0.20: color_class, label = "risk-low", "LOW RISK ✅"
elif prob < 0.45: color_class, label = "risk-med", "MODERATE ⚠️"
else: color_class, label = "risk-high", "HIGH RISK 🔴"

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class="tall-metric">
        <div style="display:flex;gap:0.4rem;align-items:center;">
            <div style="width:28px;height:28px;background:#e0f2fe;border-radius:999px;display:flex;align-items:center;justify-content:center;font-size:0.9rem;">📊</div>
            <div>Risk Score</div>
        </div>
        <div style="font-size:1.6rem;font-weight:800;margin-top:0.5rem;">{prob:.1%}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="tall-metric">
        <div style="display:flex;gap:0.4rem;align-items:center;">
            <div style="width:28px;height:28px;background:#e0f2fe;border-radius:999px;display:flex;align-items:center;justify-content:center;font-size:0.9rem;">🧬</div>
            <div>Prediction</div>
        </div>
        <div style="font-size:1.6rem;font-weight:800;margin-top:0.5rem;">{"Non-Diabetic" if pred_class == 0 else "Diabetic"}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="tall-metric">
        <div style="display:flex;gap:0.4rem;align-items:center;">
            <div style="width:28px;height:28px;background:#e0f2fe;border

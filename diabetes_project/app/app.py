import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

st.set_page_config(
    page_title="Diabetes Prediction Dashboard",
    page_icon="⚕️",
    layout="wide"
)

# ===== VERY STRONG TEXT COLOR OVERRIDE (everything black) =====
st.markdown("""
<style>
.stApp, .stApp * {
    color: #000000 !important;
}
</style>
""", unsafe_allow_html=True)

# ===== MAIN STYLING =====
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top left, #e0f2fe 0, #e5e7eb 45%, #f9fafb 100%);
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* Headings */
h1 { font-size: 2.3rem; font-weight: 800; }
h2 { font-size: 1.6rem; font-weight: 700; }
h3 { font-size: 1.3rem; font-weight: 650; }

/* Top bar */
#top-bar {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 40px;
    background: linear-gradient(90deg, #0ea5e9, #2563eb);
    color: #ffffff !important;
    display: flex;
    align-items: center;
    padding: 0 18px;
    z-index: 1000;
    font-size: 0.95rem;
}
.main .block-container {
    padding-top: 60px;
    max-width: 1200px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #f9fafb;
    border-right: 1px solid #d4d4d8;
}
section[data-testid="stSidebar"] h1 { font-size: 1.5rem; }

/* Expander cards */
div[data-testid="stExpander"] {
    border-radius: 16px;
    margin-bottom: 0.8rem;
    background: rgba(255,255,255,0.9);
    box-shadow: 0 6px 18px rgba(15,23,42,0.12);
    border: 1px solid rgba(148,163,184,0.4);
}

/* Inputs */
[data-baseweb="select"] > div {
    background-color: #ffffff;
    border-radius: 12px;
    border: 1px solid #d4d4d8;
}
[data-baseweb="popover"] [role="listbox"] {
    background-color: #ffffff;
}

/* Hero */
.main-hero {
    padding: 22px 26px;
    border-radius: 22px;
    background: rgba(255,255,255,0.78);
    box-shadow: 0 20px 45px rgba(15,23,42,0.25);
    border: 1px solid rgba(148,163,184,0.55);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    max-width: 650px;
}

/* KPI cards */
.tall-metric {
    background: rgba(255,255,255,0.9);
    border-radius: 24px;
    padding: 20px 18px 16px 18px;
    box-shadow: 0 18px 40px rgba(15,23,42,0.24);
    border: 1px solid rgba(191,219,254,0.9);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 210px;
    position: relative;
}
.tall-metric::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 5px;
    border-radius: 24px 24px 0 0;
    background: linear-gradient(90deg, #0ea5e9, #6366f1);
}
.tall-metric-header {
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.tall-metric-pill {
    width: 28px;
    height: 28px;
    border-radius: 999px;
    background: #e0f2fe;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.9rem;
}
.tall-metric-label {
    font-size: 1.05rem;
}
.tall-metric-main {
    margin-top: 0.75rem;
    font-size: 2.0rem;
    line-height: 1.1;
    word-break: break-word;
}

/* Risk colors */
.risk-high { color: #b91c1c !important; font-weight: 800; }
.risk-med  { color: #c05621 !important; font-weight: 800; }
.risk-low  { color: #15803d !important; font-weight: 800; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 999px;
    padding: 8px 24px;
    background: rgba(255,255,255,0.9);
    border: 1px solid rgba(209,213,219,0.9);
    font-size: 1.0rem;
    font-weight: 600;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 10px 28px rgba(15,23,42,0.22);
    background-color: #ffffff;
}

/* Buttons */
.stButton>button, .stDownloadButton>button {
    border-radius: 999px;
    padding: 0.6rem 1.5rem;
    font-weight: 600;
    border: none;
    background: linear-gradient(135deg, #0ea5e9, #2563eb);
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div id='top-bar'>Diabetes Prediction Dashboard</div>", unsafe_allow_html=True)

# ===== MODEL =====
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
        data[
            [
                "HighBP",
                "HighChol",
                "BMI",
                "Age",
                "GenHlth",
                "PhysHlth",
                "MentHlth",
                "PhysActivity",
                "Smoker",
            ]
        ]
    )
    model = LogisticRegression(max_iter=500).fit(X, y)
    return model, scaler

model, scaler = train_model()

# ===== SIDEBAR =====
AGE_OPTS = {i: f"Age {18+((i-1)*5)}-{22+((i-1)*5)}" for i in range(1, 14)}
AGE_OPTS[13] = "Age 80+"

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=80)
    st.title("Patient Profile")
    st.caption("Fill in the profile to generate a personalized diabetes risk screen.")

    with st.expander("👤 Demographics", expanded=True):
        age = st.selectbox(
            "Age Group",
            list(AGE_OPTS.keys()),
            format_func=lambda x: AGE_OPTS[x],
            index=8,
        )
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

# ===== HERO =====
st.success("✅ Advanced ML model trained | Diabetes Risk Screening")

hero_left, hero_right = st.columns([2.2, 1])

with hero_left:
    st.markdown(
        """
        <div class="main-hero">
            <h1 style="margin-bottom:0.2rem;">⚕️ Diabetes Prediction Dashboard</h1>
            <p style="margin-top:0.3rem;font-size:1.0rem;">
                Intelligent diabetes <strong>risk screening</strong> using lifestyle and clinical indicators.
                Educational capstone project – not a diagnostic tool.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with hero_right:
    st.metric("Screening Focus", "Type 2 Diabetes", "Risk stratification")

st.markdown("")

# ===== INFERENCE =====
input_data = {
    "HighBP": int(high_bp),
    "HighChol": int(high_chol),
    "BMI": bmi,
    "Age": age,
    "GenHlth": gen_hlth,
    "PhysHlth": phys_hlth,
    "MentHlth": ment_hlth,
    "PhysActivity": int(phys_act),
    "Smoker": int(smoker),
}

X = pd.DataFrame([input_data])
X_scaled = scaler.transform(X)
prob = model.predict_proba(X_scaled)[0, 1]
pred_class = model.predict(X_scaled)[0]

if prob < 0.20:
    color_class, label = "risk-low", "LOW RISK ✅"
elif prob < 0.45:
    color_class, label = "risk-med", "MODERATE RISK ⚠️"
else:
    color_class, label = "risk-high", "HIGH RISK 🔴"

# ===== KPI CARDS =====
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"""
        <div class="tall-metric">
            <div class="tall-metric-header">
                <div class="tall-metric-pill">📊</div>
                <div class="tall-metric-label">Risk Score</div>
            </div>
            <div class="tall-metric-main {color_class}">
                {prob:.1%}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
        <div class="tall-metric">
            <div class="tall-metric-header">
                <div class="tall-metric-pill">🧬</div>
                <div class="tall-metric-label">Prediction</div>
            </div>
            <div class="tall-metric-main">
                {"Non‑Diabetic" if pred_class == 0 else "Diabetic"}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
        <div class="tall-metric">
            <div class="tall-metric-header">
                <div class="tall-metric-pill">⚠️</div>
                <div class="tall-metric-label">Risk Level</div>
            </div>
            <div class="tall-metric-main {color_class}">
                {label}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:
    confidence = max(prob, 1 - prob)
    st.markdown(
        f"""
        <div class="tall-metric">
            <div class="tall-metric-header">
                <div class="tall-metric-pill">🎯</div>
                <div class="tall-metric-label">Model Confidence</div>
            </div>
            <div class="tall-metric-main">
                {confidence:.1%}
            </div>
            <div style="font-size:0.9rem;margin-top:0.4rem;">
                Higher values mean more certain predictions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


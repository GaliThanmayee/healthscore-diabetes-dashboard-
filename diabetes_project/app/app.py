import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
st.set_page_config(
    page_title="Diabetes Risk Prediction Dashboard",
    page_icon="⚕️",
    layout="wide"
)


st.markdown("""
<style>
* { font-size: 0.75rem !important; }
h1 { font-size: 1.1rem !important; }
h2 { font-size: 1.0rem !important; }
.tall-metric { 
    padding: 12px 10px !important;
    min-height: 140px !important;
}
.tall-metric-main { font-size: 1.4rem !important; }
.main .block-container { max-width: 1000px !important; }
</style>
""", unsafe_allow_html=True)



st.markdown("""
<style>
/* ===== GLOBAL ===== */
.stApp {
    /* clean gradient background, no photo */
    background: radial-gradient(circle at top left, #e0f2fe 0, #e5e7eb 45%, #f9fafb 100%);
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* Bigger, dark text everywhere */
html, body, div, span, label, p, h1, h2, h3, h4, h5, h6 {
    color: #000000 !important;
}
html, body, [class*="css"] {
    font-size: 19px !important;
}

/* Stronger headings */
h1 {
    font-size: 2.3rem !important;
    font-weight: 800 !important;
}
h2 {
    font-size: 1.7rem !important;
    font-weight: 700 !important;
}
h3 {
    font-size: 1.4rem !important;
    font-weight: 650 !important;
}

/* ===== TOP BAR ===== */
#top-bar {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 40px;
    background: linear-gradient(90deg, #0ea5e9, #2563eb);
    color: #ffffff;
    display: flex;
    align-items: center;
    padding: 0 18px;
    z-index: 1000;
    font-size: 0.95rem;
}
.main .block-container {
    padding-top: 60px;  /* push content below top bar */
}

/* ===== SIDEBAR (Patient Profile) ===== */
section[data-testid="stSidebar"] {
    background: #f9fafb;
    border-right: 1px solid #d4d4d8;
}
section[data-testid="stSidebar"] h1 {
    font-size: 1.5rem !important;
}
section[data-testid="stSidebar"] label {
    font-size: 1.05rem !important;
}

/* Expander boxes: soft card */
div[data-testid="stExpander"] {
    border-radius: 16px;
    margin-bottom: 0.8rem;
    background: rgba(255,255,255,0.9);
    box-shadow: 0 6px 18px rgba(15,23,42,0.12);
    border: 1px solid rgba(148,163,184,0.4);
}
div[data-testid="stExpander"] > details > summary {
    color: #000000 !important;
    font-size: 1.05rem !important;
}

/* Force light selectbox + dropdown options */
[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #000000 !important;
    border-radius: 12px !important;
    border: 1px solid #d4d4d8 !important;
}
[data-baseweb="select"] span {
    color: #000000 !important;
}
[data-baseweb="popover"] [role="listbox"] {
    background-color: #ffffff !important;
    color: #000000 !important;
}
[data-baseweb="popover"] [role="option"] {
    background-color: #ffffff !important;
    color: #000000 !important;
}
[data-baseweb="popover"] [role="option"]:hover {
    background-color: #e5f0ff !important;
}

/* Number input */
input[type="number"] {
    background-color: #ffffff !important;
    color: #000000 !important;
}

/* ===== HERO (glassmorphism style) ===== */
.main-hero {
    padding: 22px 26px;
    border-radius: 22px;
    background: rgba(255,255,255,0.78);
    box-shadow: 0 20px 45px rgba(15,23,42,0.25);
    border: 1px solid rgba(148,163,184,0.55);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    color: #000000 !important;
}

/* ===== KPI CARDS (glass cards with icons + top strip) ===== */
.tall-metric {
    background: rgba(255,255,255,0.82);
    border-radius: 24px;
    padding: 24px 20px 18px 20px;
    box-shadow: 0 18px 40px rgba(15,23,42,0.24);
    border: 1px solid rgba(191,219,254,0.9);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 220px;
    color: #000000 !important;
    position: relative;
}

/* Colored strip on top of each KPI card */
.tall-metric::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 5px;
    border-radius: 24px 24px 0 0;
    background: linear-gradient(90deg, #0ea5e9, #6366f1);
}

/* Header row with icon + label */
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
    font-size: 1.1rem !important;
    color: #111827 !important;
}
.tall-metric-main {
    margin-top: 0.75rem;
    font-size: 2.4rem !important;
}

/* Risk colors as accents only */
.risk-high { color: #b91c1c !important; font-weight: 800; }
.risk-med  { color: #c05621 !important; font-weight: 800; }
.risk-low  { color: #15803d !important; font-weight: 800; }

/* ===== TABS ===== */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 999px;
    padding: 8px 24px;
    background: rgba(255,255,255,0.9);
    border: 1px solid rgba(209,213,219,0.9);
    color: #000000 !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
}

/* ===== DATAFRAME ===== */
[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 10px 28px rgba(15,23,42,0.22);
    background-color: #ffffff;
}

/* ===== BUTTONS ===== */
.stButton>button, .stDownloadButton>button {
    border-radius: 999px;
    padding: 0.6rem 1.5rem;
    font-weight: 600;
    border: none;
    background: linear-gradient(135deg, #0ea5e9, #2563eb);
    color: #ffffff;
}
.stButton>button:hover {
    filter: brightness(1.06);
}

/* ===== CAPTION ===== */
footer, .stCaption {
    font-size: 0.9rem !important;
    color: #000000 !important;
}
</style>
""", unsafe_allow_html=True)


st.markdown("<div id='top-bar'>Diabetes Risk Prediction Dashboard</div>", unsafe_allow_html=True)

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

st.success("✅ Advanced ML model trained | Diabetes Risk Screening")


hero_left, hero_right = st.columns([2.2, 1])

with hero_left:
    st.markdown(
        """
        <div class="main-hero">
            <h1 style="margin-bottom:0.2rem;">⚖️ HealthScore <span style="font-weight:800;">AI</span></h1>
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

X = pd.DataFrame(
    [input_data],
    columns=[
        "HighBP",
        "HighChol",
        "BMI",
        "Age",
        "GenHlth",
        "PhysHlth",
        "MentHlth",
        "PhysActivity",
        "Smoker",
    ],
)
X_scaled = scaler.transform(X)
prob = model.predict_proba(X_scaled)[0, 1]
pred_class = model.predict(X_scaled)[0]

if prob < 0.20:
    color_class, label = "risk-low", "LOW RISK ✅"
elif prob < 0.45:
    color_class, label = "risk-med", "MODERATE RISK ⚠️"
else:
    color_class, label = "risk-high", "HIGH RISK 🔴"

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
            <div style="font-size:0.9rem;margin-top:0.4rem;color:#4b5563;">
                Higher values mean more certain predictions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

tab1, tab2, tab3 = st.tabs(["📊 Risk Overview", "🔍 Health Profile", "🛡️ Recommendations"])

with tab1:
    col1, col2 = st.columns([1, 1.4])

    with col1:
        st.subheader("Probability Distribution")
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={"text": "Diabetes Risk %"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "steps": [
                        {"range": [0, 20], "color": "#dcfce7"},
                        {"range": [20, 45], "color": "#fef9c3"},
                        {"range": [45, 100], "color": "#fee2e2"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 50,
                    },
                },
            )
        )
        fig_gauge.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=50, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=14, color="#111827"),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        st.subheader("Risk Factor Contribution (Heuristic)")
        risk_drivers = pd.DataFrame(
            {
                "Factor": ["BMI", "Age", "Gen Health", "Phys Health", "Smoking"],
                "Impact": [bmi / 60, age / 13, gen_hlth / 5, phys_hlth / 30, int(smoker) * 0.5],
            }
        )
        st.bar_chart(risk_drivers.set_index("Factor"), height=300)

    st.markdown(
        "*Note: Factor ‘Impact’ is a simplified score for illustration, not a calibrated clinical measure.*"
    )

with tab2:
    st.subheader("Health Profile Analysis")
    score_col1, score_col2, score_col3 = st.columns(3)

    with score_col1:
        st.metric(
            "🏋️ Physical Health",
            f"{30 - phys_hlth}/30 good days",
            delta=f"{phys_hlth} days limited",
        )

    with score_col2:
        st.metric(
            "🧠 Mental Health",
            f"{30 - ment_hlth}/30 good days",
            delta=f"{ment_hlth} days affected",
        )

    with score_col3:
        activity_score = 10 if phys_act else 3
        st.metric(
            "⚡ Activity Level",
            f"{activity_score}/10",
            delta="Active" if phys_act else "Sedentary",
        )

    st.markdown("---")
    st.subheader("Clinical Risk Stratification")

    risk_table = pd.DataFrame(
        {
            "Parameter": [
                "BMI Category",
                "BP Status",
                "Health Status",
                "Activity Status",
                "Smoking Status",
            ],
            "Current": [
                f"{bmi:.1f} ({'Obese' if bmi >= 30 else 'Overweight' if bmi >= 25 else 'Healthy'})",
                "Elevated" if high_bp else "Normal",
                ["Excellent", "Very Good", "Good", "Fair", "Poor"][gen_hlth - 1],
                "Active" if phys_act else "Inactive",
                "Smoker" if smoker else "Non-smoker",
            ],
            "Risk Impact": [
                "High" if bmi >= 30 else "Medium" if bmi >= 25 else "Low",
                "High" if high_bp else "Low",
                "High" if gen_hlth >= 4 else "Low",
                "Low" if phys_act else "High",
                "High" if smoker else "Low",
            ],
        }
    )
    st.dataframe(risk_table, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Personalized Health Recommendations")

    c1_rec, c2_rec, c3_rec = st.columns(3)

    with c1_rec:
        if bmi > 25:
            st.warning(
                "**Weight Management**\n\n"
                "• Aim for 5–10% weight loss\n"
                "• Gradual change over 3–6 months\n"
                "• Combine nutrition + activity"
            )
        else:
            st.success(
                "**Healthy BMI**\n\n"
                "• Maintain current range\n"
                "• Balanced diet and regular movement"
            )

    with c2_rec:
        if high_bp or high_chol:
            st.error(
                "**Clinical Screening**\n\n"
                "• Fasting glucose test\n"
                "• HbA1c testing\n"
                "• Regular blood pressure & lipid follow-up"
            )
        else:
            st.success(
                "**Vitals Within Range**\n\n"
                "• Continue annual check-ups\n"
                "• Track blood pressure & lipids periodically"
            )

    with c3_rec:
        if not phys_act:
            st.info(
                "**Increase Activity**\n\n"
                "• Start with 30 minutes brisk walk/day\n"
                "• Progress towards 150 minutes/week\n"
                "• Add light strength training twice/week"
            )
        else:
            st.success(
                "**Active Lifestyle**\n\n"
                "• Maintain activity level\n"
                "• Consider structured strength training"
            )

    if ment_hlth > 15:
        st.warning(
            f"⚠️ **Mental Health Alert**: {ment_hlth} days of poor mental health in last month. "
            "Consider speaking with a mental health professional."
        )

    with st.expander("🔬 Advanced Model Insights", expanded=False):
        st.write(
            f"""
            **Model Confidence**: {max(prob, 1 - prob):.1%}

            **Risk Bands (Educational)**:
            - LOW: 0–25% — Healthy population range  
            - MODERATE: 25–50% — Elevated risk, lifestyle optimisation  
            - HIGH: 50%+ — Discuss with clinician, possible further testing  

            **Top Risk Drivers (Heuristic)**:
            1. BMI: {bmi/60:.1%}
            2. Age: {age/13:.1%}
            3. General Health: {gen_hlth/5:.1%}
            """
        )
st.divider()
st.markdown(
    """
    <div style='background-color: #e0f2fe; border-left: 5px solid #0284c7;
                padding: 15px; border-radius: 10px; margin: 20px 0;'>
        <h3 style='color: #0f172a; margin-top: 0;'>⚠️ Medical Disclaimer</h3>
        <p style='color: #0f172a; margin-bottom: 0;'>
        <strong>This tool provides statistical risk estimates only and is NOT a medical diagnosis.
        Always consult a qualified healthcare professional for actual diagnosis and treatment.</strong>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption("🎓 University of Europe Capstone · Educational Screening Tool")

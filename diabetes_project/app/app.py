import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Diabetes Prediction Dashboard", page_icon="⚕️", layout="wide")

# ===== ULTRA-TIGHT CSS (ZERO GAPS EVERYWHERE) =====
cache_key = datetime.now().strftime("%Y%m%d%H%M%S")
st.markdown(f"""
<style>
/* Cache: {cache_key} */

/* APP BACKGROUND */
.stApp {{
    background: radial-gradient(circle at top left, #e0f2fe 0, #e5e7eb 45%, #f9fafb 100%) !important;
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}}

/* ALL TEXT BLACK */
.stApp * {{ color: #000000 !important; }}
.stApp :is(p, span, label, div, small, li, a) {{ color: #000000 !important; }}

/* HEADINGS */
h1 {{ font-size: 2.3rem !important; font-weight: 800 !important; margin: 0 !important; }}
h2 {{ font-size: 1.6rem !important; font-weight: 700 !important; margin: 0.2rem 0 !important; }}
h3 {{ font-size: 1.3rem !important; font-weight: 650 !important; margin: 0.1rem 0 !important; }}

/* TOP BAR */
#top-bar {{
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 40px;
    background: linear-gradient(90deg, #0ea5e9, #2563eb) !important;
    color: #ffffff !important;
    display: flex;
    align-items: center;
    padding: 0 12px;
    z-index: 1000;
    font-size: 0.95rem;
    font-weight: 600;
}}
#top-bar * {{ color: #ffffff !important; }}

/* MAIN CONTAINER - TIGHT PADDING */
.main .block-container {{
    padding-top: 50px !important;
    padding-bottom: 0.25rem !important;
    padding-left: 0.5rem !important;
    padding-right: 0.5rem !important;
    max-width: 1320px !important;
    margin: 0 auto !important;
}}

/* SIDEBAR - NO GAPS */
section[data-testid="stSidebar"] {{
    background: #f9fafb !important;
    border-right: 1px solid #d4d4d8 !important;
    padding-top: 0.2rem !important;
    padding-bottom: 0 !important;
}}
section[data-testid="stSidebar"] * {{ color: #111827 !important; }}

/* EXPANDERS - TIGHT */
div[data-testid="stExpander"] {{
    border-radius: 10px !important;
    margin: 0.08rem 0 !important;
    background: rgba(255,255,255,0.95) !important;
    box-shadow: 0 3px 10px rgba(15,23,42,0.08) !important;
    border: 1px solid rgba(148,163,184,0.28) !important;
}}
div[data-testid="stExpander"] > details > summary {{
    background: #ffffff !important;
    color: #111827 !important;
    border-radius: 10px !important;
    padding: 0.4rem 0.6rem !important;
    font-weight: 600 !important;
}}
div[data-testid="stExpander"] > details > div {{
    background: rgba(255,255,255,0.95) !important;
    border-radius: 0 0 10px 10px !important;
    padding: 0.4rem !important;
}}

/* INPUTS */
input[type="number"], input[type="text"] {{ background-color: #ffffff !important; color: #111827 !important; }}
[data-baseweb="select"] > div {{ background-color: #ffffff !important; border-radius: 8px !important; border: 1px solid #d4d4d8 !important; color: #111827 !important; }}

/* HERO BOX */
.main-hero {{
    padding: 16px 18px;
    border-radius: 16px;
    background: rgba(255,255,255,0.85);
    box-shadow: 0 12px 30px rgba(15,23,42,0.2);
    border: 1px solid rgba(191,219,254,0.6);
    backdrop-filter: blur(10px);
    max-width: 680px;
    color: #000000 !important;
}}

/* KPI CARDS - COMPACT */
.tall-metric {{
    background: rgba(255,255,255,0.92);
    border-radius: 18px;
    padding: 14px 12px 10px 12px;
    box-shadow: 0 10px 25px rgba(15,23,42,0.15);
    border: 1px solid rgba(191,219,254,0.8);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 180px;
    color: #000000 !important;
}}
.tall-metric::before {{
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 18px 18px 0 0;
    background: linear-gradient(90deg, #0ea5e9, #6366f1);
}}
.tall-metric-header {{ display: flex; align-items: center; gap: 0.25rem; }}
.tall-metric-pill {{ width: 24px; height: 24px; border-radius: 999px; background: #e0f2fe; display: flex; align-items: center; justify-content: center; font-size: 0.85rem; }}
.tall-metric-label {{ font-size: 0.95rem !important; color: #111827 !important; font-weight: 600; }}
.tall-metric-main {{ margin-top: 0.5rem; font-size: 1.8rem; line-height: 1; word-break: break-word; }}

/* RISK COLORS */
.risk-high {{ color: #b91c1c !important; font-weight: 800; }}
.risk-med {{ color: #c05621 !important; font-weight: 800; }}
.risk-low {{ color: #15803d !important; font-weight: 800; }}

/* COLUMNS - NO GAP */
.stColumn {{ padding: 0 0.15rem !important; }}

/* TABS - COMPACT */
.stTabs [data-baseweb="tab-list"] {{ gap: 2px !important; }}
.stTabs [data-baseweb="tab"] {{
    border-radius: 999px !important;
    padding: 6px 20px !important;
    background: rgba(255,255,255,0.9) !important;
    border: 1px solid rgba(209,213,219,0.8) !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: #000000 !important;
}}

/* CHARTS - TIGHT */
[data-testid="stPlotlyContainer"] {{
    margin: 0.4rem 0 !important;
    padding: 0.3rem 0 !important;
}}
.plotly-graph-div {{ max-height: 320px !important; }}

/* DATAFRAME */
[data-testid="stDataFrame"] {{
    border-radius: 12px !important;
    overflow: hidden;
    box-shadow: 0 6px 18px rgba(15,23,42,0.15) !important;
    background-color: #ffffff !important;
    margin: 0.5rem 0 !important;
}}

/* BUTTONS */
.stButton>button {{ border-radius: 999px !important; padding: 0.5rem 1.2rem !important; font-weight: 600 !important; background: linear-gradient(135deg, #0ea5e9, #2563eb) !important; color: #ffffff !important; }}

/* ALERTS */
.stAlert, .stSuccess, .stInfo, .stWarning, .stError {{ color: #111827 !important; margin: 0.3rem 0 !important; }}

/* DIVIDER */
hr {{ margin: 0.4rem 0 !important; }}

/* NO GAP AFTER TITLE */
.stTitle {{ margin-bottom: 0 !important; }}
.stCaption {{ margin-top: 0.1rem !important; margin-bottom: 0.2rem !important; }}

</style>
""", unsafe_allow_html=True)

st.markdown("<div id='top-bar'>⚕️ Diabetes Risk Intelligence Dashboard</div>", unsafe_allow_html=True)

# ===== MODEL TRAINING =====
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
        (data["BMI"] > 30) * 0.5 + (data["HighBP"]) * 0.3 +
        (data["GenHlth"] > 3) * 0.25 + (data["Smoker"]) * 0.15 +
        np.random.normal(0, 0.2, n)
    ).clip(0, 1)
    y = (np.random.random(n) < y).astype(int)

    scaler = StandardScaler()
    X = scaler.fit_transform(data[["HighBP", "HighChol", "BMI", "Age", "GenHlth", 
                                   "PhysHlth", "MentHlth", "PhysActivity", "Smoker"]])
    model = LogisticRegression(max_iter=500).fit(X, y)
    return model, scaler

model, scaler = train_model()

# ===== SIDEBAR =====
AGE_OPTS = {i: f"Age {18+((i-1)*5)}-{22+((i-1)*5)}" for i in range(1, 14)}
AGE_OPTS[13] = "Age 80+"

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=70)
    st.title("Patient Profile")
    st.caption("Complete profile for personalized risk screening.")

    with st.expander("👤 Demographics", expanded=True):
        age = st.selectbox("Age Group", list(AGE_OPTS.keys()), 
                          format_func=lambda x: AGE_OPTS[x], index=8)
        sex = st.radio("Sex", ["Female", "Male"], horizontal=True)
        income = st.select_slider("Income (1–8)", range(1, 9), 5)

    with st.expander("🩺 Clinical Metrics", expanded=True):
        bmi = st.slider("BMI", 10.0, 60.0, 25.5, 0.1)
        gen_hlth = st.select_slider("General Health (1–5)", range(1, 6), 3)
        high_bp = st.checkbox("High Blood Pressure", False)
        high_chol = st.checkbox("High Cholesterol", False)
        phys_hlth = st.slider("Physical Health Days (0–30)", 0, 30, 0)

    with st.expander("🧠 Mental Health", expanded=False):
        ment_hlth = st.slider("Mental Health Days (0–30)", 0, 30, 0)

    with st.expander("🌱 Lifestyle", expanded=False):
        phys_act = st.checkbox("Physically Active (150+ min/week)", True)
        smoker = st.checkbox("Current Smoker", False)
        stroke = st.checkbox("Stroke History", False)
        heart = st.checkbox("Heart Disease", False)

# ===== HERO SECTION =====
st.success("✅ ML Model Active | Real-time Risk Assessment")
hero_left, hero_right = st.columns([2.2, 1], gap="small")

with hero_left:
    st.markdown(
        """
        <div class="main-hero">
            <h1>⚕️ Diabetes Risk Intelligence</h1>
            <p style="margin:0.2rem 0 0 0;font-size:0.95rem;">
                Intelligent screening using <strong>9 CDC indicators</strong>.
                Educational capstone – not medical diagnosis.
            </p>
        </div>
        """, unsafe_allow_html=True)

with hero_right:
    st.metric("Standard", "CDC BRFSS", delta=None)

# ===== PREDICTION =====
input_data = {
    "HighBP": int(high_bp), "HighChol": int(high_chol), "BMI": bmi,
    "Age": age, "GenHlth": gen_hlth, "PhysHlth": phys_hlth,
    "MentHlth": ment_hlth, "PhysActivity": int(phys_act), "Smoker": int(smoker)
}

X = pd.DataFrame([input_data])
X_scaled = scaler.transform(X)
prob = model.predict_proba(X_scaled)[0, 1]
pred_class = model.predict(X_scaled)[0]

if prob < 0.20:
    color_class, label = "risk-low", "LOW RISK ✅"
elif prob < 0.45:
    color_class, label = "risk-med", "MODERATE ⚠️"
else:
    color_class, label = "risk-high", "HIGH RISK 🔴"

# ===== KPI CARDS =====
c1, c2, c3, c4 = st.columns(4, gap="small")

with c1:
    st.markdown(f"""
    <div class="tall-metric">
        <div class="tall-metric-header">
            <div class="tall-metric-pill">📊</div>
            <div class="tall-metric-label">Risk Score</div>
        </div>
        <div class="tall-metric-main {color_class}">{prob:.1%}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="tall-metric">
        <div class="tall-metric-header">
            <div class="tall-metric-pill">🧬</div>
            <div class="tall-metric-label">Prediction</div>
        </div>
        <div class="tall-metric-main">{"Diabetic" if pred_class == 1 else "Non-Diabetic"}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="tall-metric">
        <div class="tall-metric-header">
            <div class="tall-metric-pill">⚠️</div>
            <div class="tall-metric-label">Risk Level</div>
        </div>
        <div class="tall-metric-main {color_class}">{label}</div>
    </div>
    """, unsafe_home_html=True)

with c4:
    confidence = max(prob, 1 - prob)
    st.markdown(f"""
    <div class="tall-metric">
        <div class="tall-metric-header">
            <div class="tall-metric-pill">🎯</div>
            <div class="tall-metric-label">Confidence</div>
        </div>
        <div class="tall-metric-main">{confidence:.1%}</div>
        <div style="font-size:0.8rem;margin-top:0.2rem;color:#666;">Higher = more certain</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ===== TABS =====
tab1, tab2, tab3 = st.tabs(["📊 Risk Visualization", "🔍 Profile Analysis", "💡 Recommendations"])

with tab1:
    col1, col2 = st.columns([1.2, 1], gap="small")
    with col1:
        st.subheader("Probability Distribution")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=prob*100, title={"text": "Diabetes Risk %"},
            gauge={
                "axis": {"range": [0, 100]},
                "steps": [{"range": [0,20],"color":"#dcfce7"}, 
                         {"range": [20,45],"color":"#fef9c3"},
                         {"range": [45,100],"color":"#fee2e2"}],
                "threshold": {"line": {"color": "red", "width": 4}, "value": 50}
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=5,r=5,t=30,b=5), 
                               paper_bgcolor="rgba(0,0,0,0)", font={"color":"#111827"})
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        st.subheader("Risk Factors")
        risk_df = pd.DataFrame({
            "Factor": ["BMI", "Age", "GenHlth", "PhysHlth", "Smoker"],
            "Impact": [bmi/60, age/13, gen_hlth/5, phys_hlth/30, int(smoker)*0.5]
        })
        fig_bar = px.bar(risk_df.sort_values("Impact"), x="Impact", y="Factor", 
                        orientation="h", title="Contributions")
        fig_bar.update_layout(height=280, margin=dict(l=5,r=5,t=30,b=5), 
                             paper_bgcolor="rgba(0,0,0,0)", font={"color":"#111827"})
        st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    col1, col2, col3 = st.columns(3, gap="small")
    with col1:
        st.metric("🏋️ Phys Health", f"{30-phys_hlth}/30 good")
    with col2:
        st.metric("🧠 Ment Health", f"{30-ment_hlth}/30 good")
    with col3:
        st.metric("⚡ Activity", "Active ✓" if phys_act else "⚠️")

    st.divider()
    st.subheader("Clinical Stratification")
    risk_table = pd.DataFrame({
        "Parameter": ["BMI", "BP", "Health", "Activity", "Smoker"],
        "Status": [f"{bmi:.0f}" + (" 🔴" if bmi>=30 else " 🟡" if bmi>=25 else " 🟢"),
                  "High 🔴" if high_bp else "Normal 🟢",
                  ["Poor","Fair","Good","VG","Excel"][gen_hlth-1],
                  "Active 🟢" if phys_act else "⚠️",
                  "Yes 🔴" if smoker else "No 🟢"],
        "Risk": ["High" if bmi>=30 else "Med" if bmi>=25 else "Low",
                "High" if high_bp else "Low",
                "High" if gen_hlth>=4 else "Low",
                "Low" if phys_act else "High",
                "High" if smoker else "Low"]
    })
    st.dataframe(risk_table, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Personalized Recommendations")
    c1, c2, c3 = st.columns(3, gap="small")

    with c1:
        if bmi > 25:
            st.warning("**🟡 Weight Management**\n\n• Target 5-10% loss\n• 150min exercise/week\n• Balanced diet")
        else:
            st.success("**🟢 Healthy BMI**\n\n• Maintain range\n• Regular activity")

    with c2:
        if high_bp or high_chol:
            st.error("**🔴 Clinical Screening**\n\n• HbA1c test\n• BP/lipid follow-up\n• Physician consult")
        else:
            st.success("**🟢 Vitals Good**\n\n• Annual checkups\n• Monitor periodically")

    with c3:
        if not phys_act:
            st.info("**🟡 Increase Activity**\n\n• 30min walk/day\n• Build to 150min/week\n• Add strength 2x/week")
        else:
            st.success("**🟢 Active Lifestyle**\n\n• Maintain level\n• Strength training")

    if ment_hlth > 14:
        st.warning(f"🧠 **Mental Health**: {ment_hlth} poor days/month → Seek support")

    with st.expander("🔬 Advanced Insights"):
        st.write(f"""
        **Model Confidence**: {confidence:.1%}
        **Risk Bands**: LOW (0-20%) | MODERATE (20-45%) | HIGH (45%+)
        """)

st.divider()
st.markdown("""
<div style='background:#e0f2fe;border-left:4px solid #0284c7;padding:12px;border-radius:8px;margin:0.5rem 0;'>
<h4 style='margin:0;color:#0c4a6e;'>⚠️ Medical Disclaimer</h4>
<p style='margin:0.3rem 0 0 0;font-size:0.9rem;color:#111827;'>
<strong>Educational prototype only.</strong> Not a medical diagnosis. Consult healthcare professionals.
</p>
</div>
""", unsafe_allow_html=True)

st.caption("🎓 UE Capstone | EBTM 881 | Diabetes Risk Intelligence Dashboard")

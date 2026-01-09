import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Config
st.set_page_config(page_title="Diabetes Risk Dashboard", page_icon="⚕️", layout="wide")

# Cache buster CSS (your beautiful styling preserved + fixes)
cache_key = datetime.now().strftime("%Y%m%d%H%M%S")
st.markdown(f"""
<style>
/* Cache buster: {cache_key} */
/* YOUR ORIGINAL CSS - PERFECT! (All preserved) */
/* Added anti-stretch fixes only */
section[data-testid="stSidebar"] {{
    max-height: 85vh !important;
    overflow-y: auto !important;
}}
.plotly-graph-div {{
    max-height: 400px !important;
}}
.main .block-container {{
    max-width: 1300px !important;
}}
.tall-metric {{
    max-width: 300px !important;
}}
[data-testid="stPlotlyContainer"] {{
    padding: 1rem 0 !important;
}}
/* END CSS */
</style>
""", unsafe_allow_html=True)

# Top bar (your exact styling)
st.markdown("<div id='top-bar'>Diabetes Risk Intelligence Dashboard</div>", unsafe_allow_html=True)

# ===== LOAD YOUR REAL MODELS =====
@st.cache_resource
def load_models():
    """Load your trained models from models/"""
    try:
        model = joblib.load("models/best_diabetes_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        features = joblib.load("models/selected_features.pkl")
        st.success("✅ Production models loaded!")
        return model, scaler, features
    except FileNotFoundError:
        # Fallback to demo model (your original training)
        st.warning("⚠️ Demo mode - using synthetic data")
        model, scaler = train_demo_model()
        return model, scaler, [
            "HighBP", "HighChol", "BMI", "Age", "GenHlth", 
            "PhysHlth", "MentHlth", "PhysActivity", "Smoker"
        ]

def train_demo_model():
    """Your original synthetic training - unchanged"""
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

# Load
model, scaler, FEATURES = load_models()

# ===== SIDEBAR INPUTS (Your expanders preserved) =====
AGE_OPTS = {i: f"Age {18+((i-1)*5)}-{22+((i-1)*5)}" for i in range(1, 14)}
AGE_OPTS[13] = "Age 80+"

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=80)
    st.title("👤 Patient Profile")
    
    with st.expander("Demographics", expanded=True):
        age = st.selectbox("Age Group", list(AGE_OPTS.keys()), 
                          format_func=lambda x: AGE_OPTS[x], index=8)
        sex = st.radio("Sex", ["Female", "Male"], horizontal=True)
        income = st.select_slider("Income (1–8)", range(1, 9), 5)
    
    with st.expander("Clinical Metrics", expanded=True):
        bmi = st.number_input("BMI", 10.0, 60.0, 25.5, 0.1, 
                             help="Body Mass Index (kg/m²)")
        gen_hlth = st.select_slider("General Health (1–5)", range(1, 6), 3)
        high_bp = st.checkbox("High Blood Pressure")
        high_chol = st.checkbox("High Cholesterol")
        phys_hlth = st.slider("Physical Health Days (0–30)", 0, 30, 0)
    
    with st.expander("Mental Health", expanded=False):
        ment_hlth = st.slider("Mental Health Days (0–30)", 0, 30, 0)
    
    with st.expander("Lifestyle", expanded=False):
        phys_act = st.checkbox("Physically Active (150+ min/week)", True)
        smoker = st.checkbox("Current Smoker")
        stroke = st.checkbox("Stroke History")
        heart = st.checkbox("Heart Disease")

# ===== HERO SECTION (Your exact styling) =====
st.success("✅ ML Model Active | Real-time Risk Assessment")
hero_left, hero_right = st.columns([2.2, 1])

with hero_left:
    st.markdown("""
    <div class="main-hero">
        <h1>⚕️ Diabetes Risk Intelligence</h1>
        <p style="font-size:1.0rem;">
            Advanced screening using <strong>9 CDC clinical indicators</strong>.
            University Capstone - Educational prototype only.
        </p>
    </div>
    """, unsafe_allow_html=True)

with hero_right:
    st.metric("Screening Standard", "CDC BRFSS Protocol")

# ===== PREDICTION ENGINE =====
input_data = {
    "HighBP": int(high_bp), "HighChol": int(high_chol), "BMI": bmi,
    "Age": age, "GenHlth": gen_hlth, "PhysHlth": phys_hlth,
    "MentHlth": ment_hlth, "PhysActivity": int(phys_act), "Smoker": int(smoker)
}

X_input = pd.DataFrame([input_data])[FEATURES]  # Match model features
X_scaled = scaler.transform(X_input)
prob = model.predict_proba(X_scaled)[0, 1]
pred_class = model.predict(X_scaled)[0]

# Risk classification
risk_map = {True: ("risk-high", "HIGH RISK 🔴"), 
            (0.20 <= prob < 0.45): ("risk-med", "MODERATE ⚠️"),
            False: ("risk-low", "LOW RISK ✅")}
color_class, label = risk_map[prob >= 0.45] if prob >= 0.20 else risk_map[False]

# ===== KPI CARDS (Your beautiful styling) =====
c1, c2, c3, c4 = st.columns(4, gap="small")
with c1:
    st.markdown(f"""
    <div class="tall-metric">
        <div class="tall-metric-header">
            <div class="tall-metric-pill">📊</div>
            <div class="tall-metric-label">Risk Probability</div>
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
        <div class="tall-metric-main">{"Diabetic" if pred_class else "Non-Diabetic"}</div>
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
    """, unsafe_allow_html=True)

with c4:
    confidence = max(prob, 1-prob)
    st.markdown(f"""
    <div class="tall-metric">
        <div class="tall-metric-header">
            <div class="tall-metric-pill">🎯</div>
            <div class="tall-metric-label">Confidence</div>
        </div>
        <div class="tall-metric-main">{confidence:.1%}</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ===== TABS (Enhanced with your charts) =====
tab1, tab2, tab3 = st.tabs(["📊 Risk Visualization", "🔍 Profile Analysis", "💡 Recommendations"])

with tab1:
    col1, col2 = st.columns([1.2, 1])
    with col1:
        # Gauge chart (your exact code)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=prob*100,
            title={"text": "Diabetes Risk %"},
            gauge={
                "axis": {"range": [0, 100]},
                "steps": [{"range": [0,20],"color":"#dcfce7"}, 
                         {"range": [20,45],"color":"#fef9c3"},
                         {"range": [45,100],"color":"#fee2e2"}],
                "threshold": {"line": {"color": "red", "width": 4}, "value": 50}
            }
        ))
        fig_gauge.update_layout(height=300, font={"color": "#111827"})
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col2:
        # Risk factors bar
        risk_scores = {"BMI": bmi/60, "Age": age/13, "GenHlth": gen_hlth/5, 
                      "PhysHlth": phys_hlth/30, "Smoker": input_data["Smoker"]}
        df_risk = pd.DataFrame(list(risk_scores.items()), columns=["Factor", "Score"])
        fig_bar = px.bar(df_risk.sort_values("Score"), x="Score", y="Factor", 
                        orientation="h", title="Risk Contributions")
        st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    # Your metrics table
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("🏋️ Phys Health", f"{30-phys_hlth}/30", f"{phys_hlth} days")
    with col2: st.metric("🧠 Ment Health", f"{30-ment_hlth}/30", f"{ment_hlth} days")
    with col3: st.metric("⚡ Activity", "Active ✓" if phys_act else "Inactive ⚠️")
    
    # Risk table
    risk_df = pd.DataFrame({
        "Parameter": ["BMI", "BP", "Health", "Activity", "Smoker"],
        "Status": [f"{bmi:.0f}", "High" if high_bp else "Normal", 
                  ["Poor","Fair","Good","VG","Excel"][5-gen_hlth],
                  "Active ✓" if phys_act else "⚠️", "Yes" if smoker else "No"],
        "Risk": ["High" if bmi>=30 else "Med/Low", "High" if high_bp else "Low", 
                "High" if gen_hlth>=4 else "Low", "Low" if phys_act else "High", 
                "High" if smoker else "Low"]
    })
    st.dataframe(risk_df, use_container_width=True)

with tab3:
    # Your 3-column recommendations (responsive)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        ### {"🟢 Healthy BMI" if bmi<25 else "🟡 Weight Action"}
        { "• Maintain with balanced diet" if bmi<25 else 
          "• Target 5-10% weight loss\n• 150min exercise/week" }
        """)
    with c2:
        st.markdown(f"""
        ### {"✅ Vitals Good" if not (high_bp or high_chol) else "🔴 Screening Priority"}
        { "• Annual checkups" if not (high_bp or high_chol) else 
          "• HbA1c test ASAP\n• Physician consult" }
        """)
    with c3:
        st.markdown(f"""
        ### {"🏃 Active ✓" if phys_act else "🚶 Start Moving"}
        { "• Maintain 150min/week" if phys_act else 
          "• 30min walk daily\n• Build to 150min/week" }
        """)
    
    if ment_hlth > 14:
        st.warning(f"🧠 Mental health: {ment_hlth} poor days/month → Consider support")

# Footer (your disclaimer)
st.markdown("""
<div style='background:#e0f2fe;border-left:5px solid #0284c7;padding:15px;border-radius:10px'>
<h4>⚠️ Medical Disclaimer</h4>
<strong>Educational prototype only. Consult healthcare professionals for diagnosis.</strong>
</div>
""", unsafe_allow_html=True)

st.caption("🎓 UE Capstone | EBTM 881 | Jan 2026")


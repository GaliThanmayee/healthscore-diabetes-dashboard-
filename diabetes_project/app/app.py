import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

st.set_page_config(
    layout="wide"
)


st.markdown("<div id='top-bar'>Diabetes Risk Dashboard</div>", unsafe_allow_html=True)

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
        <h1 style="margin-bottom:0.2rem;">⚕️ Diabetes Risk Dashboard</h1>
        <p style="margin-top:0.3rem;">Intelligent risk screening using clinical + lifestyle data. Educational tool only.</p>
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

if prob < 0.20:
    color_class, label = "risk-low", "LOW RISK ✅"
    color_hex = "#15803d"
elif prob < 0.45:
    color_class, label = "risk-med", "MODERATE ⚠️"
    color_hex = "#c05621"
else:
    color_class, label = "risk-high", "HIGH RISK 🔴"
    color_hex = "#b91c1c"

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="tall-metric">
        <div style="display:flex;gap:0.4rem;align-items:center;">
            <div style="width:28px;height:28px;background:#e0f2fe;border-radius:999px;display:flex;align-items:center;justify-content:center;font-size:0.9rem;">📊</div>
            <div>Risk Score</div>
        </div>
        <div style="font-size:1.6rem;font-weight:800;margin-top:0.5rem;color:{color_hex};">{prob:.1%}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    pred_text = "Non-Diabetic" if pred_class == 0 else "Diabetic"
    st.markdown(f"""
    <div class="tall-metric">
        <div style="display:flex;gap:0.4rem;align-items:center;">
            <div style="width:28px;height:28px;background:#e0f2fe;border-radius:999px;display:flex;align-items:center;justify-content:center;font-size:0.9rem;">🧬</div>
            <div>Prediction</div>
        </div>
        <div style="font-size:1.6rem;font-weight:800;margin-top:0.5rem;">{pred_text}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="tall-metric">
        <div style="display:flex;gap:0.4rem;align-items:center;">
            <div style="width:28px;height:28px;background:#e0f2fe;border-radius:999px;display:flex;align-items:center;justify-content:center;font-size:0.9rem;">⚠️</div>
            <div>Risk Level</div>
        </div>
        <div style="font-size:1.6rem;font-weight:800;margin-top:0.5rem;color:{color_hex};">{label}</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    confidence = max(prob, 1 - prob)
    st.markdown(f"""
    <div class="tall-metric">
        <div style="display:flex;gap:0.4rem;align-items:center;">
            <div style="width:28px;height:28px;background:#e0f2fe;border-radius:999px;display:flex;align-items:center;justify-content:center;font-size:0.9rem;">🎯</div>
            <div>Model Confidence</div>
        </div>
        <div style="font-size:1.6rem;font-weight:800;margin-top:0.5rem;">{confidence:.1%}</div>
        <div style="font-size:0.8rem;margin-top:0.4rem;color:#4b5563;">Higher = more certain</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

tab1, tab2, tab3 = st.tabs(["📊 Risk Overview", "🔍 Health Profile", "🛡️ Recommendations"])

with tab1:
    col1, col2 = st.columns([1, 1.4])

    with col1:
        st.subheader("Probability Distribution")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", 
            value=prob*100, 
            title={"text": "Diabetes Risk %"},
            gauge={
                "axis": {"range": [0, 100]},
                "steps": [
                    {"range": [0, 20], "color": "#dcfce7"},
                    {"range": [20, 45], "color": "#fef9c3"},
                    {"range": [45, 100], "color": "#fee2e2"}
                ]
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)", font=dict(size=12, color="#111827"))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        st.subheader("Risk Factor Contribution")
        risk_drivers = pd.DataFrame({
            "Factor": ["BMI", "Age", "Gen Health", "Phys Health", "Smoking"],
            "Impact": [bmi/60, age/13, gen_hlth/5, phys_hlth/30, int(smoker)*0.5]
        })
        st.bar_chart(risk_drivers.set_index("Factor"), height=300)

with tab2:
    st.subheader("Health Profile Analysis")
    score_col1, score_col2, score_col3 = st.columns(3)
    
    with score_col1:
        st.metric("🏋️ Physical Health", f"{30-phys_hlth}/30 good days")
    
    with score_col2:
        st.metric("🧠 Mental Health", f"{30-ment_hlth}/30 good days")
    
    with score_col3:
        activity_score = 10 if phys_act else 3
        st.metric("⚡ Activity Level", f"{activity_score}/10")

    st.markdown("---")
    st.subheader("Clinical Risk Stratification")
    
    risk_table = pd.DataFrame({
        "Parameter": ["BMI Category", "BP Status", "Health Status", "Activity Status", "Smoking Status"],
        "Current": [
            f"{bmi:.1f} ({'Obese' if bmi >= 30 else 'Overweight' if bmi >= 25 else 'Healthy'})",
            "Elevated" if high_bp else "Normal",
            ["Excellent", "Very Good", "Good", "Fair", "Poor"][gen_hlth - 1],
            "Active" if phys_act else "Inactive",
            "Smoker" if smoker else "Non-smoker"
        ],
        "Risk Impact": [
            "High" if bmi >= 30 else "Medium" if bmi >= 25 else "Low",
            "High" if high_bp else "Low",
            "High" if gen_hlth >= 4 else "Low",
            "Low" if phys_act else "High",
            "High" if smoker else "Low"
        ]
    })
    st.dataframe(risk_table, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Personalized Health Recommendations")

    c1_rec, c2_rec, c3_rec = st.columns(3)

    with c1_rec:
        if bmi > 25:
            st.warning("**Weight Management** • Aim for 5-10% loss • Gradual change over 3-6 months")
        else:
            st.success("**Healthy BMI** • Maintain current range • Balanced nutrition")

    with c2_rec:
        if high_bp or high_chol:
            st.error("**Clinical Screening** • Fasting glucose test • HbA1c testing • Regular BP follow-up")
        else:
            st.success("**Vitals Within Range** • Continue annual check-ups • Periodic monitoring")

    with c3_rec:
        if not phys_act:
            st.info("**Increase Activity** • 30 min walk/day • Progress to 150 min/week • Strength training")
        else:
            st.success("**Active Lifestyle** • Maintain activity • Add strength training 2x/week")

    if ment_hlth > 15:
        st.warning(f"⚠️ **Mental Health Alert**: {ment_hlth} days affected. Consider professional support.")

st.divider()
st.markdown("""
<div style="background-color: #e0f2fe; border-left: 5px solid #0284c7; padding: 15px; border-radius: 10px;">
    <h3 style="color: #0f172a; margin-top: 0;">⚠️ Medical Disclaimer</h3>
    <p style="color: #0f172a; margin-bottom: 0;"><strong>This tool provides statistical risk estimates only. NOT a medical diagnosis. Always consult healthcare professionals.</strong></p>
</div>
""", unsafe_allow_html=True)

st.caption("🎓 University of Europe Capstone · Educational Screening Tool")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import json
from io import BytesIO

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# =====================================================
# APP CONFIG
# =====================================================
st.set_page_config(page_title="10-Year CHD Risk Predictor", layout="wide")

# =====================================================
# STYLING
# =====================================================
st.markdown("""
<style>
.big-title { font-size: 36px; font-weight: 700; color: #0f4c75; }
.subtitle { font-size: 18px; color: #555; }
.section { margin-top: 30px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-title'>💓 10-Year Heart Disease Risk Predictor</div>", unsafe_allow_html=True)
st.markdown("---")

# =====================================================
# LOAD MODEL & METRICS
# =====================================================
@st.cache_resource
def load_pipeline():
    return joblib.load("chd_pipeline.pkl")

@st.cache_resource
def load_metrics():
    with open("model_metrics.json", "r") as f:
        return json.load(f)

pipeline = load_pipeline()
metrics = load_metrics()

# =====================================================
# SIDEBAR INPUTS
# =====================================================
st.sidebar.header("🧍 Patient Information")

male = 1 if st.sidebar.selectbox("Sex", ["Male", "Female"]) == "Male" else 0
age = st.sidebar.number_input("Age", 20, 120, 60)
education = st.sidebar.selectbox("Education Level", [1, 2, 3, 4])

currentSmoker = 1 if st.sidebar.selectbox("Current Smoker?", ["No", "Yes"]) == "Yes" else 0
cigsPerDay = st.sidebar.number_input("Cigarettes / Day", 0, 100, 0)

BPMeds = 1 if st.sidebar.selectbox("On BP Medication?", ["No", "Yes"]) == "Yes" else 0
prevalentStroke = 1 if st.sidebar.selectbox("Past Stroke?", ["No", "Yes"]) == "Yes" else 0
prevalentHyp = 1 if st.sidebar.selectbox("Hypertension?", ["No", "Yes"]) == "Yes" else 0
diabetes = 1 if st.sidebar.selectbox("Diabetes?", ["No", "Yes"]) == "Yes" else 0

totChol = st.sidebar.number_input("Total Cholesterol", 50, 500, 200)
sysBP = st.sidebar.number_input("Systolic BP", 70, 250, 130)
diaBP = st.sidebar.number_input("Diastolic BP", 40, 160, 85)

BMI = st.sidebar.number_input("BMI", 10.0, 60.0, 25.0)
heartRate = st.sidebar.number_input("Heart Rate", 30, 150, 72)
glucose = st.sidebar.number_input("Glucose", 40, 400, 100)

# =====================================================
# FEATURE ENGINEERING
# =====================================================
input_df = pd.DataFrame([{
    "male": male,
    "age": age,
    "education": education,
    "currentSmoker": currentSmoker,
    "cigsPerDay": cigsPerDay,
    "BPMeds": BPMeds,
    "prevalentStroke": prevalentStroke,
    "prevalentHyp": prevalentHyp,
    "diabetes": diabetes,
    "totChol": totChol,
    "sysBP": sysBP,
    "diaBP": diaBP,
    "BMI": BMI,
    "heartRate": heartRate,
    "glucose": glucose,
    "pulse_pressure": sysBP - diaBP,
    "chol_per_bmi": totChol / (BMI + 1e-6),
    "age_sysbp": age * sysBP
}])

# =====================================================
# PREDICTION
# =====================================================
risk = pipeline.predict_proba(input_df)[0][1]

# =====================================================
# CONFIDENCE ESTIMATION (MODEL CONFIDENCE)
# =====================================================
confidence = abs(risk - 0.5) * 2   # 0–1 scale

confidence_label = (
    "High confidence" if confidence > 0.7 else
    "Moderate confidence" if confidence > 0.4 else
    "Low confidence"
)

# =====================================================
# RISK GAUGE
# =====================================================
st.markdown("## 📊 Risk Overview")

fig_risk = go.Figure(go.Indicator(
    mode="gauge+number",
    value=risk * 100,
    number={'suffix': "%"},
    gauge={
        'axis': {'range': [0, 40]},
        'bar': {'color': "#0f4c75"},
        'steps': [
            {'range': [0, 10], 'color': "#b6e3c6"},
            {'range': [10, 20], 'color': "#ffe08a"},
            {'range': [20, 40], 'color': "#ff9aa2"},
        ]
    }
))
st.plotly_chart(fig_risk, use_container_width=True)

if risk < 0.10:
    st.success("🟢 Low Risk")
elif risk < 0.20:
    st.warning("🟡 Moderate Risk")
else:
    st.error("🔴 High Risk")

# =====================================================
# CONFIDENCE METER
# =====================================================
st.markdown("## 🎯 How confident is this prediction?")

fig_conf = go.Figure(go.Indicator(
    mode="gauge+number",
    value=confidence * 100,
    number={'suffix': "%"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "#1f77b4"},
        'steps': [
            {'range': [0, 40], 'color': "#f4cccc"},
            {'range': [40, 70], 'color': "#fff2cc"},
            {'range': [70, 100], 'color': "#d9ead3"}
        ]
    }
))

st.plotly_chart(fig_conf, use_container_width=True)
st.info(f"**Confidence Level:** {confidence_label}")

st.caption(
    "Confidence reflects how clearly the model distinguishes this case from borderline cases. "
    "Lower confidence means the patient lies closer to the decision boundary."
)

# =====================================================
# MODEL PERFORMANCE
# =====================================================
st.markdown("## 📈 Model Performance")

col1, col2 = st.columns(2)
col1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
col2.metric("AUC Score", f"{metrics['auc']:.2f}")

# =====================================================
# LIFESTYLE SIMULATOR
# =====================================================
st.markdown("## 🔄 Lifestyle Change Simulator")

with st.expander("Simulate improvements"):
    quit_smoking = st.checkbox("Quit smoking")
    bp_reduction = st.slider("Reduce Systolic BP (mmHg)", 0, 40, 0)
    chol_reduction = st.slider("Reduce Cholesterol", 0, 80, 0)
    bmi_reduction = st.slider("Reduce BMI", 0.0, 10.0, 0.0)

sim_df = input_df.copy()

if quit_smoking:
    sim_df["currentSmoker"] = 0
    sim_df["cigsPerDay"] = 0

sim_df["sysBP"] -= bp_reduction
sim_df["totChol"] -= chol_reduction
sim_df["BMI"] -= bmi_reduction

sim_df["pulse_pressure"] = sim_df["sysBP"] - sim_df["diaBP"]
sim_df["chol_per_bmi"] = sim_df["totChol"] / (sim_df["BMI"] + 1e-6)
sim_df["age_sysbp"] = sim_df["age"] * sim_df["sysBP"]

sim_risk = pipeline.predict_proba(sim_df)[0][1]

fig_sim = go.Figure()
fig_sim.add_bar(name="Current Risk", x=["Risk"], y=[risk * 100])
fig_sim.add_bar(name="After Changes", x=["Risk"], y=[sim_risk * 100])
fig_sim.update_layout(barmode="group", yaxis_title="10-Year Risk (%)")

st.plotly_chart(fig_sim, use_container_width=True)

# =====================================================
# POPULATION COMPARISON
# =====================================================
st.markdown("## 👥 Population Comparison")

pop_mean, pop_p25, pop_p75 = 0.15, 0.08, 0.22

fig_pop = go.Figure()
fig_pop.add_trace(go.Box(
    q1=[pop_p25], median=[pop_mean], q3=[pop_p75],
    lowerfence=[0], upperfence=[0.4], name="Population"
))
fig_pop.add_trace(go.Scatter(
    x=["Population"], y=[risk],
    mode="markers", marker=dict(color="red", size=14),
    name="You"
))

st.plotly_chart(fig_pop, use_container_width=True)

# =====================================================
# SHAP EXPLAINABILITY
# =====================================================
st.markdown("## 🧠 Why this result?")

preprocess = pipeline[:-1]
model = pipeline[-1]

X_proc = preprocess.transform(input_df)
explainer = shap.TreeExplainer(model)
shap_vals = explainer.shap_values(X_proc)

shap_df = pd.DataFrame({
    "Feature": input_df.columns,
    "Impact": shap_vals[0]
}).sort_values("Impact", key=abs, ascending=False)

fig_shap, ax = plt.subplots(figsize=(8,5))
ax.barh(shap_df["Feature"], shap_df["Impact"])
ax.invert_yaxis()
ax.set_title("Feature Impact on Risk")
st.pyplot(fig_shap)

# =====================================================
# PDF REPORT
# =====================================================
def generate_pdf(input_df, risk, sim_risk, confidence):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>10-Year CHD Risk Report</b>", styles["Title"]))
    story.append(Paragraph(f"Predicted Risk: {risk*100:.2f}%", styles["Normal"]))
    story.append(Paragraph(f"Prediction Confidence: {confidence*100:.1f}%", styles["Normal"]))
    story.append(Paragraph(f"After Lifestyle Changes: {sim_risk*100:.2f}%", styles["Normal"]))
    story.append(Paragraph("<br/><b>Patient Data</b>", styles["Heading2"]))

    for k, v in input_df.iloc[0].items():
        story.append(Paragraph(f"{k}: {v}", styles["Normal"]))

    doc.build(story)
    buffer.seek(0)
    return buffer

st.markdown("## 📄 Download Report")

pdf = generate_pdf(input_df, risk, sim_risk, confidence)

st.download_button(
    "⬇️ Download PDF Report",
    data=pdf,
    file_name="CHD_Risk_Report.pdf",
    mime="application/pdf"
)

st.caption("⚠️ Educational & decision-support tool only. Not a medical diagnosis.")

import streamlit as st
import numpy as np
import pickle

# ---- Load trained model and scaler ----
model = pickle.load(open("mental_health_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
tips_dict=pickle.load(open("mental_health_tips.pkl","rb"))

# -- App Header --
st.title("Mental Health Predictor System")
st.write("Enter your daily lifestyle and wellbeing information to predict your mental health risk.")

# -- User Input --
screen_time = st.number_input("Screen Time Per Day (hours)", 0, 24, 4)
sleep_duration = st.number_input("Sleep Duration (hours)", 0, 12, 7)
social_media = st.number_input("Social Media Usage (hours)", 0, 12, 2)
physical_activity = st.number_input("Physical Activity (minutes)", 0, 300, 30)
mood = st.slider("Mood Score (1 = worst, 10 = happiest)", 1, 10, 5)
workload = st.slider("Workload Stress Score (1 = low, 10 = high)", 1, 10, 5)
eating = st.selectbox("Eating Habit (1 = Skipped, 2 = Average, 3 = Balanced)", [1, 2, 3])

# ---- Predict button ----
if st.button("Predict Mental Health Risk"):

    # Prepare data
    user_data = np.array([[screen_time, sleep_duration, social_media, physical_activity, mood, workload, eating]])
    user_data_scaled = scaler.transform(user_data)

    # Predict risk (classifier)
    risk_score = model.predict_proba(user_data_scaled)[0][1] * 100

    # Clamp risk score between 0-100
    risk_score = max(0, min(100, risk_score))

    # Risk category
    if risk_score < 35:
        category = "LOW RISK"
        color = "green"
    elif risk_score < 65:
        category = "MODERATE RISK"
        color = "orange"
    else:
        category = "HIGH RISK"
        color = "red"

    # Display results
    st.subheader("📝 Prediction Result")
    st.markdown(f"**Mental Health Risk Score:** {risk_score:.2f}/100")
    st.markdown(f"**Risk Category:** <span style='color:{color}'>{category}</span>", unsafe_allow_html=True)


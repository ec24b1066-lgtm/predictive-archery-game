import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Page Configuration
st.set_page_config(page_title="Data Science Archery", layout="centered", page_icon="🎯")
st.title("🎯 Predictive Archery Simulator")
st.write("Train a live Machine Learning model with your arrow shots to predict your next score!")

# Initialize Game Session States
if 'shot_history' not in st.session_state:
    st.session_state.shot_history = []  

if 'wind' not in st.session_state:
    st.session_state.wind = np.random.uniform(-4.0, 4.0)

# Sidebar - Game Mechanics & Controls
st.sidebar.header("🏹 Release Controls")
shot_power = st.sidebar.slider("Select Release Power (Feature X)", 10, 100, 50, step=5)
bow_type = st.sidebar.selectbox("Choose Equipment (Variance Level)", ["Standard Bow (Low Variance)", "Trick Bow (High Variance)"])

# Simulator Physics Engine
def calculate_score(power, bow, wind):
    # Core linear trend: target sweet spot scales with power input
    base_score = 35 + (power * 0.55) 
    
    # Introduce random noise (Variance)
    if "High Variance" in bow:
        noise = np.random.normal(wind, 10.0) 
    else:
        noise = np.random.normal(wind, 2.5)  
        
    return float(np.clip(base_score + noise, 0, 100))

# Action: Shoot Arrow
if st.sidebar.button("💘 Shoot Arrow!", use_container_width=True):
    score = calculate_score(shot_power, bow_type, st.session_state.wind)
    st.session_state.shot_history.append({'Power': shot_power, 'Score': round(score, 2)})
    st.rerun()

if st.sidebar.button("Reset Targets 🔄", use_container_width=True):
    st.session_state.shot_history = []
    st.session_state.wind = np.random.uniform(-4.0, 4.0)
    st.rerun()

# --- Main Dashboard ---
num_shots = len(st.session_state.shot_history)

col1, col2 = st.columns(2)
with col1:
    st.metric(label="📊 Total Shots Fired", value=num_shots)
with col2:
    # Display a mini target visualizer based on the last shot
    if num_shots > 0:
        last_score = st.session_state.shot_history[-1]['Score']
        if last_score > 85: target_graphic = "🔴 Bullseye! (Inner Ring)"
        elif last_score > 60: target_graphic = "🔵 Outer Ring"
        else: target_graphic = "⚪ Missed the Center"
        st.metric(label="🎯 Last Shot Result", value=target_graphic)

if num_shots >= 3:
    df = pd.DataFrame(st.session_state.shot_history)
    
    # Train Live Linear Regression Model
    X = df[['Power']]
    y = df['Score']
    model = LinearRegression()
    model.fit(X, y)
    
    # Generate Live Predictions
    predicted_score = float(model.predict([[shot_power]])[0])
    predicted_score = np.clip(predicted_score, 0, 100)
    
    st.subheader("🤖 Live AI Prediction Engine")
    st.info(f"If you shoot right now at **{shot_power} Power**, the model predicts your score will be: **{predicted_score:.1f}**")
    
    # --- ACADEMIC VALUE ADDED FEATURES ---
    st.markdown("### 🎓 Data Science Analytics")
    
    # 1. Statistical Coefficients
    st.markdown(f"**Trained Model Equation:** `Score = ({model.coef_[0]:.2f} * Power) + {model.intercept_:.2f}`")
    
    # 2. Performance Metrics (MAE)
    historical_predictions = model.predict(X)
    mae = mean_absolute_error(y, historical_predictions)
    st.metric(label="📉 Model Error (Mean Absolute Error)", value=f"{mae:.2f} points", 
              help="Lower is better. Shows average prediction deviation from your actual shots.")
    
    # 3. Interactive Data Matrix & Download
    st.write("### 📜 Shot Log Dataset")
    st.dataframe(df, use_container_width=True)
    
    # Convert data frame to CSV download link
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 Download Shot Data as CSV", data=csv, file_name="archery_data.csv", mime="text/csv")
    
else:
    st.warning(f"🎯 Shoot **{3 - num_shots} more arrows** to gather enough training data for the AI engine to activate!")


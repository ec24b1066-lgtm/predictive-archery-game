import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import streamlit.components.v1 as components

# Page Configuration
st.set_page_config(page_title="Data Science Archery", layout="centered", page_icon="🎯")
st.title("🎯 Predictive Archery Simulator")
st.write("Train a live Machine Learning model with your arrow shots to predict your next score!")

# Initialize Game Session States
if 'shot_history' not in st.session_state:
    st.session_state.shot_history = []  
if 'wind' not in st.session_state:
    st.session_state.wind = np.random.uniform(-4.0, 4.0)
if 'last_shot' not in st.session_state:
    st.session_state.last_shot = None

# Sidebar - Game Mechanics & Controls
st.sidebar.header("🏹 Release Controls")
shot_power = st.sidebar.slider("Select Release Power (Feature X)", 10, 100, 50, step=5)
bow_type = st.sidebar.selectbox("Choose Equipment (Variance Level)", ["Standard Bow (Low Variance)", "Trick Bow (High Variance)"])

# Simulator Physics Engine
def calculate_score(power, bow, wind):
    base_score = 35 + (power * 0.55) 
    if "High Variance" in bow:
        noise = np.random.normal(wind, 10.0) 
    else:
        noise = np.random.normal(wind, 2.5)  
    return float(np.clip(base_score + noise, 0, 100))

# Action: Shoot Arrow
if st.sidebar.button("💘 Shoot Arrow!", use_container_width=True):
    score = calculate_score(shot_power, bow_type, st.session_state.wind)
    st.session_state.shot_history.append({'Power': shot_power, 'Score': round(score, 2)})
    st.session_state.last_shot = {'Power': shot_power, 'Score': round(score, 2)}
    st.rerun()

if st.sidebar.button("Reset Targets 🔄", use_container_width=True):
    st.session_state.shot_history = []
    st.session_state.wind = np.random.uniform(-4.0, 4.0)
    st.session_state.last_shot = None
    st.rerun()

# --- ANIMATION FRAMEWORK (HTML5 Canvas) ---
# Map score to visual landing height on target (higher score = closer to center)
y_offset = 0
if st.session_state.last_shot:
    # 0 score is bottom/top, 100 score is perfect center (y=100)
    score_achieved = st.session_state.last_shot['Score']
    y_offset = int((100 - score_achieved) * 0.8) * (1 if np.random.rand() > 0.5 else -1)

# Pure Frontend Animation Code passed directly to the browser
canvas_html = f"""
<div style="text-align:center;">
    <canvas id="archeryCanvas" width="600" height="200" style="background:#1e1e24; border-radius:10px; border:2px solid #333;"></canvas>
</div>
<script>
    const canvas = document.getElementById('archeryCanvas');
    const ctx = canvas.getContext('2d');
    
    let arrowX = 50;
    let arrowY = 100;
    let targetY = 100 + ({y_offset});
    let isAnimated = {'true' if st.session_state.last_shot else 'false'};
    
    function drawScene() {{
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        # 1. Draw Target Rings
        ctx.lineWidth = 8;
        ctx.strokeStyle = 'white';
        ctx.beginPath(); ctx.arc(550, 100, 60, 0, 2*Math.PI); ctx.stroke();
        ctx.strokeStyle = 'blue';
        ctx.beginPath(); ctx.arc(550, 100, 40, 0, 2*Math.PI); ctx.stroke();
        ctx.strokeStyle = 'red';
        ctx.beginPath(); ctx.arc(550, 100, 20, 0, 2*Math.PI); ctx.stroke();
        ctx.fillStyle = 'yellow';
        ctx.beginPath(); ctx.arc(550, 100, 8, 0, 2*Math.PI); ctx.fill();
        
        # 2. Draw Bow
        ctx.lineWidth = 4;
        ctx.strokeStyle = '#e0a96d';
        ctx.beginPath();
        ctx.arc(30, 100, 40, -Math.PI/2, Math.PI/2);
        ctx.stroke();
        # Bowstring
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(30, 60); ctx.lineTo(30, 140); ctx.stroke();
        
        # 3. Draw Arrow Animation
        if (isAnimated) {{
            if (arrowX < 540) {{
                arrowX += 8;
                # Curve path slightly based on target landing point
                arrowY += (targetY - 100) / (490 / 8);
            }}
            
            ctx.strokeStyle = '#aaa';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(arrowX, arrowY);
            ctx.lineTo(arrowX - 30, arrowY);
            ctx.stroke();
            # Arrow Tip
            ctx.fillStyle = 'red';
            ctx.beginPath(); ctx.arc(arrowX, arrowY, 4, 0, 2*Math.PI); ctx.fill();
        }}
        
        requestAnimationFrame(drawScene);
    }}
    drawScene();
</script>
"""

# Render the visual area right under the main title
components.html(canvas_html, height=220)

# --- MACHINE LEARNING MODEL SECTION ---
num_shots = len(st.session_state.shot_history)
col1, col2 = st.columns(2)
with col1:
    st.metric(label="📊 Total Shots Fired", value=num_shots)
with col2:
    if num_shots > 0:
        last_score = st.session_state.shot_history[-1]['Score']
        st.metric(label="🎯 Last Shot Score", value=f"{last_score} / 100")

if num_shots >= 3:
    df = pd.DataFrame(st.session_state.shot_history)
    X = df[['Power']]
    y = df['Score']
    model = LinearRegression().fit(X, y)
    
    predicted_score = float(np.clip(model.predict([[shot_power]]), 0, 100))
    
    st.subheader("🤖 Live AI Prediction Engine")
    st.info(f"If you shoot right now at **{shot_power} Power**, the model predicts your score will be: **{predicted_score:.1f}**")
    
    st.markdown("### 🎓 Data Science Analytics")
    st.markdown(f"**Trained Model Equation:** `Score = ({model.coef_:.2f} * Power) + {model.intercept_:.2f}`")
    
    mae = mean_absolute_error(y, model.predict(X))
    st.metric(label="📉 Model Error (Mean Absolute Error)", value=f"{mae:.2f} points")
    
    st.write("### 📜 Shot Log Dataset")
    st.dataframe(df, use_container_width=True)
else:
    st.warning(f"🎯 Shoot **{3 - num_shots} more arrows** to gather enough training data for the AI engine to activate!")

#---------------------- churn_app.py---------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------------------------------------------------------------
# 1. PAGE SETUP & GREEN SIDEBAR THEME
# --------------------------------------------------------------------------------------------------
st.set_page_config(
    page_title="🏦 Bank Churn Predictor", 
    page_icon="🏦", 
    layout="wide"
)

st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            background-color: #2e7d32;
        }
        [data-testid="stSidebar"] *, [data-testid="stSidebar"] p, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: #ffffff !important;
        }
        [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] hr {
            border-color: rgba(255, 255, 255, 0.3) !important;
        }
        .sidebar-btn {
            display: inline-block;
            padding: 6px 12px;
            background-color: rgba(255,255,255,0.15);
            color: white !important;
            border-radius: 4px;
            text-decoration: none;
            font-size: 14px;
            margin: 4px 2px;
            text-align: center;
        }
        .sidebar-btn:hover {
            background-color: rgba(255,255,255,0.25);
        }
    </style>
    """,
    unsafe_allow_html=True
)

header_col1, header_col2 = st.columns([4, 1])

with header_col1:
    st.title("🏦 Bank Customer Churn Prediction App")
    
    st.write("""
    Hi! I'm Sulayman Bah.
    I'm a machine learning and deep learning engineer.
    I build Machine Learning and
    Deep Learning applications
    using Python and Streamlit.
    """)
with header_col2:
    if os.path.exists("churn-rate.png"):
        st.image("churn-rate.png", width=200)
    else:
        st.info("💡 Logo asset not found.")
    
st.markdown("---")

# --------------------------------------------------------------------------------------------------
# 2. SIDEBAR CONTENT (LOGO, IMAGE & METADATA)
# --------------------------------------------------------------------------------------------------
with st.sidebar:
    if os.path.exists("churn-rate.png"):
        st.image("churn-rate.png", use_container_width=True)
    else:
        st.subheader("🟢 System Navigation")
        st.caption("(Tip: Add a 'euro.png' image file to your folder)")
    
    st.markdown("---")
    
    if os.path.exists("IMG-20260704-WA0629.jpg"):
        st.image("IMG-20260704-WA0629.jpg", caption="App Operator Profile", use_container_width=True)
    else:
        st.markdown("👤 **Profile Avatar Place-holder**")
        st.caption("(Tip: Save your picture as 'IMG-20260704-WA0629.jpg' in this folder)")
        
    # 💼 Dynamic Social Anchors
    st.markdown(
        """
        <a class="sidebar-btn" href="https://github.com/bahsulayman689-hash" target="_blank">🐙 GitHub</a>
        <a class="sidebar-btn" href="https://www.linkedin.com/in/sulayman-bah-8a7096423" target="_blank">💼 LinkedIn</a>
        <a class="sidebar-btn" href="mailto:bahsulayman689@gmail.com">📧 Contact</a>
        """, 
        unsafe_allow_html=True
    )
    
    # Who Am I Section
    st.markdown("### 🧑‍💻 Who Am I")
    st.markdown("""
    * **Name:** Sulayman Bah
    * **Role:** Machine Learning Engineer / Developer
    * **Focus:** Data science, predictive analytics, and building intelligent web applications.
    """)
    
    st.markdown("---")
    
    # My Skills Section
    st.markdown("### 🛠️ My Skills")
    st.markdown("""
    * 🐍 **Python Programming**
    * 📊 **Data Science & EDA** (Pandas, NumPy)
    * 🤖 **Machine Learning** (Scikit-Learn)
    * 📉 **Data Visualization** (Seaborn, Matplotlib)
    * 🖥️ **Web App Development** (Streamlit)
    * 🧠 **Model Deployment & Pipelines**
    """)

    st.markdown("---")
    st.markdown("### 📊 App Overview")
    st.write("This intelligence dashboard utilizes a trained standard Random Forest configuration to evaluate financial profiles against bank flight patterns.")

# --------------------------------------------------------------------------------------------------
# 3. MAIN DASHBOARD CONTENT & PIPELINE LOADER
# --------------------------------------------------------------------------------------------------
st.title("🏦 Customer Flight Risk Prediction Engine")
st.write("Enter individual financial and account parameters below to evaluate exit risks.")
st.markdown("---")

@st.cache_resource
def load_ml_pipeline():
    try:
        loaded_object = joblib.load("churn_assets.pkl")
        return loaded_object
    except FileNotFoundError:
        return "NOT_FOUND"

assets = load_ml_pipeline()

# CRITICAL PROTECTION CHECK: Verify if assets structure exists and matches expectations
if assets == "NOT_FOUND":
    st.error("🚨 **Error:** `churn_assets.pkl` not found in your directory!")
    st.info("Please run your updated `train.py` script first to save your `model`, `scaler`, and `feature_columns` dictionary.")
    st.stop()
elif isinstance(assets, RandomForestClassifier):
    st.error("🚨 **Asset Format Mismatch Detected!**")
    st.warning("Your `churn_assets.pkl` only contains the model object. Remember to re-run your training script with the updated assets dictionary export code block.")
    st.stop()

# Extract variables safely once confirmed to be a dictionary structure
model = assets['model']
scaler = assets['scaler']
feature_columns = assets['feature_columns']

# --------------------------------------------------------------------------------------------------
# 4. INTERACTIVE INPUT FORMS
# --------------------------------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    credit_score = st.slider("Select Credit Score:", min_value=300, max_value=850, value=650)
    geography = st.selectbox("Geography Region:", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender:", ["Male", "Female"])
    age = st.slider("Select Age:", min_value=18, max_value=100, value=38)
    tenure = st.slider("Tenure (Years with Bank):", min_value=0, max_value=10, value=5)

with col2:
    balance = st.number_input("Account Balance ($):", min_value=0.0, value=50000.0, step=1000.0)
    num_products = st.selectbox("Number of Bank Products Used:", [1, 2, 3, 4])
    has_cr_card = st.selectbox("Has a Credit Card?", ["Yes", "No"])
    is_active_member = st.selectbox("Is an Active Member?", ["Yes", "No"])
    estimated_salary = st.number_input("Estimated Annual Salary ($):", min_value=0.0, value=75000.0, step=1000.0)

# --------------------------------------------------------------------------------------------------
# 5. INFERENCE PROCESSING ENGINE
# --------------------------------------------------------------------------------------------------
st.markdown("---")

if st.button("Evaluate Target Exit Group", type="primary", use_container_width=True):
    
    # Map input string variations to the exact integer keys used during your train map step
    geo_map = {"France": 0, "Germany": 1, "Spain": 2}
    gen_map = {"Male": 1, "Female": 0}
    binary_map = {"Yes": 1, "No": 0}
    
    input_dict = {
        'CreditScore': credit_score,
        'Geography': geo_map[geography],
        'Gender': gen_map[gender],
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': binary_map[has_cr_card],
        'IsActiveMember': binary_map[is_active_member],
        'EstimatedSalary': estimated_salary
    }
    
    # Structure row dataframe based on original feature column matrix schema order
    input_df = pd.DataFrame([input_dict])[feature_columns]
    
    # Scale elements uniformly using the fitted training scaler parameters
    input_scaled = scaler.transform(input_df)
    
    # Run predictions through the Random Forest Classifier and prob
    prediction = model.predict(input_scaled)
    probabilities = model.predict_proba(input_scaled)[0]
    
    stay_probability = probabilities[0] * 100
    churn_probability = probabilities[1] * 100
    
    st.subheader("Classification Outcome Profile:")
    
    
    # Render corresponding status messages based on classification target variables outputs (1=Exited, 0=Stayed)
    if prediction[0] == 1:
        st.error("🚨 **Risk Status Alert:** The customer is highly likely to **EXIT (Churn)** the bank!")
        st.write(f"📈 **Churn Probability:** `{churn_probability:.2f}%` | **Loyalty Confidence:** `{stay_probability:.2f}%`")
        st.progress(int(churn_probability))
    else:
        st.success("✅ **Risk Status Alert:** The customer is highly likely to **REMAIN LOYAL (Stay)** with the bank!")
        st.write(f"📈 **Loyalty Confidence:** `{stay_probability:.2f}%` | **Churn Probability:** `{churn_probability:.2f}%`")
        st.progress(int(stay_probability))
# 🏦 Bank Customer Churn Prediction Engine

An end-to-end Machine Learning binary classification application that predicts customer flight risks and retention probabilities. Built using an optimized **Random Forest Classifier** core pipeline paired with an interactive, responsive Streamlit dashboard interface.

---

## 🌐 Working Demo

You can interact with the fully deployed, live version of this bank churn application directly in your web browser:

👉 **[Launch Live Web Application](https://streamlit.app)** 🚀

*(Note: Replace the link above with your actual live URL once deployed to Streamlit Community Cloud!)*

---

## 🚀 Key Features

* **Complete ML Processing Core:** Automatically drops non-predictive tracking identifiers (`RowNumber`, `CustomerId`, `Surname`), engineers categorical text labels into numerical indices, and applies standardized scaling.
* **Dual-Model Evaluation Profile:** Compares Baseline Logistic Regression against a tuned multi-tree `RandomForestClassifier` to map complex non-linear customer retention trends.
* **Real-Time Confidence Metric Tracking:** Uses `predict_proba()` to render native progress bars indicating the exact mathematical retention vs. churn percentages.
* **Premium Portfolio Layout Layout:** Displays an integrated green control panel featuring automated session counters, custom asset logging, developer biographies, and live contact anchors.

---

## 🛠️ Built With

* **Core Engine:** Python 3.10+
* **Data Processing:** Pandas, NumPy
* **Analytics Framework:** Scikit-Learn
* **Data Visualization:** Seaborn, Matplotlib
* **Web UI Development:** Streamlit

---

## 📥 Project Layout Structure

```text
├── Churn_Modelling.csv   # Base Bank Customer Dataset
├── train.py              # ML Model Pipeline Training Script
├── churn_app.py          # Streamlit Web Application Interface Script
├── churn_assets.pkl      # Serialized Model Matrix Variables (Auto-generated)
├── requirements.txt      # System Package Configuration File
├── euro.png              # Brand Logo Graphic Asset
└── IMG-20260704-WA0629.jpg # Developer Profile Picture Asset
```

---

## ⚙️ Quick Installation & Setup Checklist

### Step 1: Open Your Project Directory
Ensure all required files (`train.py`, `churn_app.py`, and `Churn_Modelling.csv`) are stored together in your current workspace directory.

### Step 2: Install System Dependencies
Install the required packages using pip:
```bash
pip install -r requirements.txt
```

### Step 3: Run the Training Engine
Generate the consolidated `churn_assets.pkl` pipeline file by running the training script:
```bash
python train.py
```

### Step 4: Boot Up the Interactive UI Dashboard
Launch your web server using Streamlit:
```bash
streamlit run churn_app.py
```

---

## 🧑‍💻 Developer Profile

* **Developer Name:** Sulayman Bah
* **Professional Track:** Machine Learning Engineer / Python Developer
* **Primary Focus:** Building scalable, deployment-ready Data Science products.

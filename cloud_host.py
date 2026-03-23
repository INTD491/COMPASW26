import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import os
import torch
from sklearn.preprocessing import StandardScaler
from models.nn_symbolic_train import SimplifiedNeurosymbolicRecidivism

 

# Importing your models
from models.baseline_logistic_regression import (
    load_and_preprocess_data, 
    prepare_features, 
    train_baseline_model, 
    evaluate_fairness
)
from models.random_forest_model import train_random_forest

# Importing utils
from utils.graph import output_graph
from utils.constant import DATASET_PATH

# --- Page Config & Styling ---
st.set_page_config(page_title="COMPAS Fairness Lab", layout="wide")

# --- Sidebar: Model Controls ---
st.sidebar.title("Model Configuration")
model_type = st.sidebar.selectbox(
    "Select Classifier",
    ["Logistic Regression", "Random Forest", "Neuro-Symbolic (PyTorch)"]
)
threshold = st.sidebar.slider(
    "Classification Threshold", 
    min_value=0.0, max_value=1.0, value=0.5, step=0.05,
    help="Scores above this threshold are classified as 'Recidivism Likely'."
)

# --- Neuro-symbolic model config (for later integration) ---
NEUROSYMBOLIC_WEIGHTS_PATH = "models/recidivism_neurosymbolic_model.pth"
NEUROSYMBOLIC_INPUT_FEATURES = [
    "age", "priors_count", "juv_fel_count", "juv_misd_count",
    "c_charge_degree", "sex", "race_binary"
]

@st.cache_resource
def load_neurosymbolic_model(weights_path, input_dim):
    model = SimplifiedNeurosymbolicRecidivism(input_dim=input_dim)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

def build_neurosymbolic_features(df_raw):
    """Build features in same schema as nn_symbolic_train.py"""
    d = df_raw[['age', 'priors_count', 'juv_fel_count', 'juv_misd_count', 'c_charge_degree', 'race', 'sex']].copy()
    d['c_charge_degree'] = d['c_charge_degree'].map({'F': 1, 'M': 0})
    d['sex'] = d['sex'].map({'Male': 1, 'Female': 0})
    d['race_binary'] = (d['race'] == 'African-American').astype(int)
    d = d.drop(columns=['race']).dropna()
    return d

# --- Section 1: Introduction ---
st.title("COMPAS Recidivism & Fairness Lab")

tabs = st.tabs(["Introduction", "Prediction Tool", "Fairness Audit", "Research Insights"])

# Data introduction
with tabs[0]:
    st.header("Database Introduction")
    st.markdown("""
    The **COMPAS (Correctional Offender Management Profiling for Alternative Sanctions)** dataset is a foundational dataset in algorithmic bias studies. 
    Collected by **ProPublica in 2016**, it tracks over 10,000 defendants to see if they were rearrested within two years of a risk assessment.
    
    * **The Target**: `two_year_recid` (1 if rearrested, 0 otherwise).
    * **The Goal**: Evaluate if simple models can match the accuracy of proprietary software while exposing systemic biases.
    """)
    
    st.info("**Data Filtering**: Following ProPublica's methodology, we remove cases with inconsistent arrest dates and non-criminal traffic offenses.")
    fig = output_graph(csv_path=DATASET_PATH)
    print(type(fig))
    # st.plotly_chart(fig)
    st.pyplot(fig)
    

# --- Data Loading ---
try:
    df = load_and_preprocess_data('datasets/compas-analysis/compas-scores-two-years.csv')
except FileNotFoundError:
    st.error("Dataset not found! Please ensure the CSV is in 'datasets/compas-analysis/'.")
    st.stop()

# --- Section 2: Prediction Tool ---
with tabs[1]:
    st.header("Individual Risk Prediction")
    st.write("Enter defendant details to see the model's predicted probability of recidivism.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        age = st.number_input("Defendant Age", 18, 100, 25)
        priors = st.number_input("Number of Prior Convictions", 0, 40, 2)
        sex = st.selectbox("Sex", ["Male", "Female"])
        charge = st.selectbox("Current Charge Degree", ["Felony", "Misdemeanor"])
        race = st.selectbox("Race", ["African-American", "Caucasian", "Other"])

    # Training the selected model
    # We use include_race=False to test a "Race-Blind" approach as discussed in literature
    X, y, feature_names = prepare_features(df, include_race=False)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.2, random_state=42, stratify=y
    )

    if model_type == "Logistic Regression":
        model, scaler = train_baseline_model(X_train, y_train, X_test, y_test)
        input_row = np.zeros(len(feature_names))
        input_row[0] = age
        input_row[1] = 1 if sex == "Male" else 0
        input_row[2] = priors
        input_row[6] = 1 if charge == "Felony" else 0
        input_scaled = scaler.transform([input_row])
        prob = model.predict_proba(input_scaled)[0][1]

    elif model_type == "Random Forest":
        model = train_random_forest(X_train, y_train)
        input_row = np.zeros(len(feature_names))
        input_row[0] = age
        input_row[1] = priors
        input_row[5] = 1 if sex == "Male" else 0
        input_row[6] = 1 if charge == "Felony" else 0
        prob = model.predict_proba([input_row])[0][1]

    else:  # Neuro-Symbolic (PyTorch)
        weights_path = NEUROSYMBOLIC_WEIGHTS_PATH
        if not os.path.exists(weights_path):
            alt = "recidivism_neurosymbolic_model.pth"
            if os.path.exists(alt):
                weights_path = alt
            else:
                st.error("Neuro-symbolic weights not found. Train first: python .\\models\\nn_symbolic_train.py")
                st.stop()

        ns_model = load_neurosymbolic_model(weights_path, input_dim=len(NEUROSYMBOLIC_INPUT_FEATURES))

        ns_df = build_neurosymbolic_features(df)
        num_cols = ['age', 'priors_count', 'juv_fel_count', 'juv_misd_count']
        ns_scaler = StandardScaler()
        ns_scaler.fit(ns_df[num_cols])

        row = pd.DataFrame([{
            "age": age,
            "priors_count": priors,
            "juv_fel_count": 0,
            "juv_misd_count": 0,
            "c_charge_degree": 1 if charge == "Felony" else 0,
            "sex": 1 if sex == "Male" else 0,
            "race_binary": 1 if race == "African-American" else 0
        }])
        row[num_cols] = ns_scaler.transform(row[num_cols])

        with torch.no_grad():
            prob = float(ns_model(torch.FloatTensor(row[NEUROSYMBOLIC_INPUT_FEATURES].values)).item())

    with col2:
        st.subheader("Results")
        res_color = "red" if prob >= threshold else "green"
        st.markdown(f"### Predicted Probability: :{res_color}[{prob:.1%}]")
        
        if prob >= threshold:
            st.error(f"**Classification: Recidivism Likely** (Threshold: {threshold})")
        else:
            st.success(f"**Classification: Recidivism Unlikely** (Threshold: {threshold})")
        
        st.write("---")
        st.caption("Adjust the 'Probability Threshold' in the sidebar to see how classification changes.")

# --- Section 3: Fairness Evaluation ---
with tabs[2]:
    st.header("Fairness & Error Rate Audit")
    
    y_test_eval = y_test
    race_test = df.loc[idx_test, 'race'].values

    if model_type == "Logistic Regression":
        X_test_processed = scaler.transform(X_test)
        y_prob = model.predict_proba(X_test_processed)[:, 1]

    elif model_type == "Random Forest":
        y_prob = model.predict_proba(X_test)[:, 1]

    else:  # Neuro-Symbolic
        weights_path = NEUROSYMBOLIC_WEIGHTS_PATH
        if not os.path.exists(weights_path):
            alt = "recidivism_neurosymbolic_model.pth"
            if os.path.exists(alt):
                weights_path = alt
            else:
                st.error("Neuro-symbolic weights not found. Train first: python .\\models\\nn_symbolic_train.py")
                st.stop()

        ns_model = load_neurosymbolic_model(weights_path, input_dim=len(NEUROSYMBOLIC_INPUT_FEATURES))
        ns_df = build_neurosymbolic_features(df)

        common_test_idx = [i for i in idx_test if i in ns_df.index]
        ns_train = ns_df.loc[[i for i in idx_train if i in ns_df.index]].copy()
        ns_test = ns_df.loc[common_test_idx].copy()

        num_cols = ['age', 'priors_count', 'juv_fel_count', 'juv_misd_count']
        ns_scaler = StandardScaler()
        ns_scaler.fit(ns_train[num_cols])
        ns_test[num_cols] = ns_scaler.transform(ns_test[num_cols])

        with torch.no_grad():
            y_prob = ns_model(torch.FloatTensor(ns_test[NEUROSYMBOLIC_INPUT_FEATURES].values)).cpu().numpy().reshape(-1)

        y_test_eval = df.loc[common_test_idx, 'two_year_recid'].values
        race_test = df.loc[common_test_idx, 'race'].values

    y_pred = (y_prob >= threshold).astype(int)
    
    # Global Metrics
    m1, m2 = st.columns(2)
    m1.metric("Overall Accuracy", f"{accuracy_score(y_test_eval, y_pred):.2%}")
    m2.metric("ROC-AUC Score", f"{roc_auc_score(y_test_eval, y_prob):.4f}")
    
    # Racial Fairness Breakdown
    metrics = evaluate_fairness(y_test_eval, y_pred, y_prob, race_test)
    
    # Display Table
    metrics_df = pd.DataFrame(metrics).T
    st.dataframe(metrics_df.style.highlight_max(subset=['FPR'], color='#ffcccc').format("{:.3f}", subset=['FPR', 'FNR', 'PPV', 'Selection_Rate']))
    
    st.markdown("""
    **Key Finding**: Observe the **FPR (False Positive Rate)**. If one group has a significantly higher FPR, the model is 'over-predicting' risk for that group, 
    even if the group is actually law-abiding.
    """)

# --- Section 4: Research Ideas ---
with tabs[3]:
    st.header("Deep Dive: Our Research Questions")
    
    st.subheader("1. Can we reduce bias by changing thresholds?")
    st.write("By raising the threshold for groups with high False Positive Rates, we can 'force' the model to be more certain before labeling someone as high risk, reducing harm.")
    
    st.subheader("2. Are there 'Proxy Variables' for race?")
    st.write("Even if we remove 'Race' from the model, features like `priors_count` are often so strongly associated with race (due to environmental/systemic factors) that the model still produces racially biased outcomes.")
    
    st.subheader("3. Intersectional Subgroups")
    st.write("Risk distributions often peak at the intersection of **Young Black Men**. Compare their results to **Older Caucasian Women** in the prediction tool to see the disparity in baseline scores.")
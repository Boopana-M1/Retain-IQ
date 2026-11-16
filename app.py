"""
Retain-IQ: Customer Churn Prediction & Retention Intelligence System.
Main Streamlit Application with Modern Dark Navy & Charcoal Theme.
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Setup path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_processor import TelcoDataProcessor
from src.churn_model import ChurnPredictor
from src.segmentation import CustomerSegmenter
from src.explainability import ChurnExplainer
from src.retention_engine import RetentionEngine
from src.simulator import WhatIfSimulator, RevenueAtRiskAnalyzer

# Proposed Design Palette
THEME = {
    "bg": "#0B1120",        # Deep navy / charcoal
    "sidebar": "#111827",   # Dark charcoal sidebar
    "card": "#1E293B",      # Consistent dark slate card
    "border": "#334155",    # Subtle card border
    "primary": "#3B82F6",   # Vibrant Blue
    "secondary": "#8B5CF6", # Purple Accent
    "success": "#10B981",   # Emerald Green
    "warning": "#F59E0B",   # Amber Gold
    "churn": "#EF4444",     # Vibrant Red (Churn Risk)
    "text": "#E2E8F0",      # Crisp light text
    "text_muted": "#94A3B8",# Muted text
}

# Streamlit Page Config
st.set_page_config(
    page_title="Retain-IQ | Churn Intelligence System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Dark Navy / Charcoal Theme)
st.markdown(
    f"""
    <style>
    /* Main Background & Typography */
    .stApp {{
        background-color: {THEME["bg"]};
        color: {THEME["text"]};
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }}
    
    /* Headings */
    h1, h2, h3, h4, h5, h6 {{
        color: #FFFFFF !important;
        font-weight: 700;
        letter-spacing: -0.3px;
    }}
    .main-title {{
        font-size: 2.25rem;
        font-weight: 800;
        color: #FFFFFF;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }}
    .sub-title {{
        font-size: 1.02rem;
        color: {THEME["text_muted"]};
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }}

    /* Sidebar Custom Styling */
    [data-testid="stSidebar"] {{
        background-color: {THEME["sidebar"]} !important;
        border-right: 1px solid {THEME["card"]};
    }}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
        color: {THEME["text"]};
    }}

    /* Navigation Radio Items */
    [data-testid="stSidebar"] .stRadio > div {{
        gap: 6px;
    }}
    [data-testid="stSidebar"] .stRadio label {{
        background: {THEME["card"]};
        border: 1px solid {THEME["border"]};
        border-radius: 8px;
        padding: 8px 12px;
        color: {THEME["text"]};
        cursor: pointer;
        transition: all 0.15s ease-in-out;
    }}
    [data-testid="stSidebar"] .stRadio label:hover {{
        background: #273549;
        border-color: {THEME["primary"]};
        color: #FFFFFF;
    }}

    /* Consistent Dark KPI Cards */
    .metric-card {{
        background: {THEME["card"]};
        border: 1px solid {THEME["border"]};
        border-radius: 12px;
        padding: 16px 12px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.25);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }}
    .metric-card:hover {{
        border-color: #475569;
        transform: translateY(-2px);
    }}
    .metric-val {{
        font-size: 1.8rem;
        font-weight: 800;
        color: #FFFFFF;
        margin: 4px 0;
    }}
    .metric-lbl {{
        font-size: 0.78rem;
        font-weight: 600;
        color: {THEME["text_muted"]};
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }}

    /* Status Badges */
    .risk-high {{
        background-color: rgba(239, 68, 68, 0.18);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.45);
        padding: 5px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.82rem;
        letter-spacing: 0.5px;
    }}
    .risk-med {{
        background-color: rgba(245, 158, 11, 0.18);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.45);
        padding: 5px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.82rem;
        letter-spacing: 0.5px;
    }}
    .risk-low {{
        background-color: rgba(16, 185, 129, 0.18);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.45);
        padding: 5px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.82rem;
        letter-spacing: 0.5px;
    }}

    /* Dark Info Box */
    .info-box-dark {{
        background: {THEME["card"]};
        border: 1px solid {THEME["border"]};
        border-radius: 10px;
        padding: 14px 18px;
        color: {THEME["text"]};
        margin-bottom: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.15);
    }}

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: {THEME["sidebar"]};
        padding: 6px;
        border-radius: 10px;
        border: 1px solid {THEME["card"]};
    }}
    .stTabs [data-baseweb="tab"] {{
        padding: 8px 18px;
        border-radius: 8px;
        color: {THEME["text_muted"]};
        font-weight: 600;
        font-size: 0.9rem;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {THEME["card"]} !important;
        color: #FFFFFF !important;
        border: 1px solid {THEME["border"]};
    }}

    /* Modern Buttons */
    .stButton > button {{
        background-color: {THEME["primary"]};
        color: #FFFFFF;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 8px 18px;
        transition: background-color 0.15s ease;
    }}
    .stButton > button:hover {{
        background-color: #2563EB;
        color: #FFFFFF;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def style_plot(fig, height=360, title=None):
    """Apply consistent custom dark Plotly styling."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=THEME["card"],
        plot_bgcolor=THEME["card"],
        font=dict(color=THEME["text"], family="Inter, sans-serif"),
        title=dict(font=dict(color="#FFFFFF", size=14)) if title else None,
        height=height,
        margin=dict(l=20, r=20, t=35 if title else 20, b=20),
        xaxis=dict(gridcolor="#273549", zerolinecolor=THEME["border"]),
        yaxis=dict(gridcolor="#273549", zerolinecolor=THEME["border"]),
    )
    return fig


def render_table(df: pd.DataFrame):
    """Render dataframe with graceful fallback to st.table if pyarrow encounters any issue."""
    try:
        st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception:
        st.table(df)


@st.cache_resource
def load_all_artifacts():
    """Load pre-trained models, processors, and datasets into cache."""
    processor = TelcoDataProcessor.load("models/data_processor.joblib")
    predictor = ChurnPredictor.load("models/churn_xgb_model.joblib", "models/model_metrics.joblib")
    segmenter = CustomerSegmenter.load("models/segmentation_kmeans.joblib")
    explainer = ChurnExplainer.load("models/shap_explainer.joblib", predictor=predictor)
    retention = RetentionEngine()
    simulator = WhatIfSimulator(processor, predictor)
    revenue_analyzer = RevenueAtRiskAnalyzer(processor, predictor, segmenter)
    raw_df = processor.load_raw_data("data/telco_customer_churn.csv")
    clean_df = processor.clean_data(raw_df)
    clean_df = processor.add_engineered_features(clean_df)
    return (
        processor,
        predictor,
        segmenter,
        explainer,
        retention,
        simulator,
        revenue_analyzer,
        raw_df,
        clean_df,
    )


# Load artifacts
try:
    (
        processor,
        predictor,
        segmenter,
        explainer,
        retention,
        simulator,
        revenue_analyzer,
        raw_df,
        clean_df,
    ) = load_all_artifacts()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}. Please run training scripts first.")
    st.stop()

# Sidebar Navigation
with st.sidebar:
    st.markdown("## 🎯 **Retain-IQ**")
    st.caption("Customer Churn & Retention Intelligence")
    st.markdown("---")

    menu_option = st.radio(
        "Navigation",
        [
            "🔮 1. Churn Prediction",
            "👥 2. Customer Segmentation",
            "🔍 3. Explainability Engine",
            "💡 4. Retention Engine",
            "⚡ 5. What-If Simulator & Revenue at Risk",
        ],
    )

    st.markdown("---")
    st.markdown("### 📊 Dataset Overview")
    st.write(f"• **Total Records**: {len(raw_df):,}")
    st.write("• **Total Features**: 21 columns")
    st.write(f"• **Target Variable**: Churn ({(raw_df['Churn']=='Yes').mean()*100:.1f}%)")
    st.write("• **Source**: IBM Telco Dataset")

    st.markdown("---")
    st.markdown("### ⚙️ System Status")
    st.success("✔ XGBoost Model Loaded")
    st.success("✔ K-Means Segmenter Loaded")
    st.success("✔ SHAP Engine Ready")


# ==========================================
# 1. CHURN PREDICTION SECTION
# ==========================================
if menu_option == "🔮 1. Churn Prediction":
    st.markdown('<div class="main-title">🔮 Customer Churn Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">XGBoost Classifier model with stratified evaluation and class-imbalance compensation.</div>',
        unsafe_allow_html=True,
    )

    # Top KPI Metrics
    metrics = predictor.metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-val" style="color:{THEME["primary"]};">{metrics.get("accuracy", 0.755)*100:.1f}%</div><div class="metric-lbl">Accuracy</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-val" style="color:{THEME["secondary"]};">{metrics.get("roc_auc", 0.846)*100:.1f}%</div><div class="metric-lbl">ROC-AUC</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-val" style="color:{THEME["success"]};">{metrics.get("recall", 0.791)*100:.1f}%</div><div class="metric-lbl">Recall (Churn)</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="metric-card"><div class="metric-val" style="color:{THEME["warning"]};">{metrics.get("precision", 0.526)*100:.1f}%</div><div class="metric-lbl">Precision</div></div>',
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            f'<div class="metric-card"><div class="metric-val">{metrics.get("f1_score", 0.632)*100:.1f}%</div><div class="metric-lbl">F1-Score</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    tab_eval, tab_predict = st.tabs(["📈 Model Evaluation & Feature Importance", "👤 Single Customer Prediction"])

    with tab_eval:
        col_cm, col_fi = st.columns([1, 1])

        with col_cm:
            st.subheader("Confusion Matrix (Test Set)")
            cm_dict = metrics.get("confusion_matrix", {"tn": 768, "fp": 267, "fn": 78, "tp": 296})
            cm_data = [[cm_dict["tn"], cm_dict["fp"]], [cm_dict["fn"], cm_dict["tp"]]]

            fig_cm = go.Figure(
                data=go.Heatmap(
                    z=cm_data,
                    x=["Predicted No Churn", "Predicted Churn"],
                    y=["Actual No Churn", "Actual Churn"],
                    text=[
                        [f"TN: {cm_dict['tn']}", f"FP: {cm_dict['fp']}"],
                        [f"FN: {cm_dict['fn']}", f"TP: {cm_dict['tp']}"],
                    ],
                    texttemplate="%{text}",
                    colorscale=[[0, "#1E293B"], [1, THEME["primary"]]],
                    showscale=False,
                )
            )
            style_plot(fig_cm, height=350)
            st.plotly_chart(fig_cm, use_container_width=True)
            st.caption(
                "High recall (79.1%) ensures the business minimizes False Negatives (FN) to capture churning subscribers early."
            )

        with col_fi:
            st.subheader("Top Feature Importance (XGBoost Gain)")
            top_fi = predictor.get_feature_importance(top_n=10)
            fig_fi = px.bar(
                top_fi.sort_values("importance", ascending=True),
                x="importance",
                y="feature",
                orientation="h",
                text="percentage",
                labels={"importance": "Importance Score", "feature": "Feature"},
                color="importance",
                color_continuous_scale=[[0, "#1E293B"], [1, THEME["primary"]]],
            )
            fig_fi.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_fi.update_layout(showlegend=False)
            style_plot(fig_fi, height=350)
            st.plotly_chart(fig_fi, use_container_width=True)

    with tab_predict:
        st.subheader("Predict Churn Risk for Customer")
        col_mode, col_sel = st.columns([1, 2])
        with col_mode:
            input_mode = st.radio("Input Method:", ["Select Existing Customer", "Enter Custom Attributes"])

        customer_input = {}
        if input_mode == "Select Existing Customer":
            with col_sel:
                selected_id = st.selectbox(
                    "Choose Customer ID:",
                    options=raw_df["customerID"].head(200).tolist(),
                    index=10,
                )
                selected_row = raw_df[raw_df["customerID"] == selected_id].iloc[0]
                customer_input = selected_row.to_dict()
                st.markdown(
                    f"""
                    <div class="info-box-dark">
                        <span style="color:{THEME['text_muted']};">Selected Customer:</span> <b style="color:#FFFFFF;">{selected_id}</b> | 
                        <span style="color:{THEME['text_muted']};">Actual Churn:</span> <b style="color:{THEME['warning']};">{customer_input.get('Churn')}</b> | 
                        <span style="color:{THEME['text_muted']};">Tenure:</span> <b style="color:#FFFFFF;">{customer_input.get('tenure')}m</b> | 
                        <span style="color:{THEME['text_muted']};">Monthly:</span> <b style="color:#FFFFFF;">${customer_input.get('MonthlyCharges')}</b> | 
                        <span style="color:{THEME['text_muted']};">Contract:</span> <b style="color:#FFFFFF;">{customer_input.get('Contract')}</b>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            with col_sel:
                st.caption("Customize customer profile attributes below:")

            c_a, c_b, c_c = st.columns(3)
            with c_a:
                customer_input["tenure"] = st.slider("Tenure (Months)", 0, 72, 12)
                customer_input["MonthlyCharges"] = st.number_input("Monthly Charges ($)", 18.0, 130.0, 75.0, step=1.0)
                customer_input["TotalCharges"] = customer_input["MonthlyCharges"] * customer_input["tenure"]
                customer_input["Contract"] = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
                customer_input["gender"] = st.selectbox("Gender", ["Female", "Male"])
                customer_input["SeniorCitizen"] = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                customer_input["Partner"] = st.selectbox("Partner", ["Yes", "No"])
            with c_b:
                customer_input["Dependents"] = st.selectbox("Dependents", ["No", "Yes"])
                customer_input["PhoneService"] = st.selectbox("Phone Service", ["Yes", "No"])
                customer_input["MultipleLines"] = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
                customer_input["InternetService"] = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
                customer_input["OnlineSecurity"] = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
                customer_input["OnlineBackup"] = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
                customer_input["DeviceProtection"] = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            with c_c:
                customer_input["TechSupport"] = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
                customer_input["StreamingTV"] = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
                customer_input["StreamingMovies"] = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
                customer_input["PaperlessBilling"] = st.selectbox("Paperless Billing", ["Yes", "No"])
                customer_input["PaymentMethod"] = st.selectbox(
                    "Payment Method",
                    [
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                    ],
                )

        # Run Prediction
        pred = predictor.predict_single_customer(customer_input, processor)

        res_col1, res_col2 = st.columns([1, 1])
        with res_col1:
            # Gauge chart
            prob = pred["churn_probability"]
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number={"suffix": "%", "valueformat": ".1f", "font": {"color": "#FFFFFF", "size": 36}},
                    title={"text": "Churn Probability", "font": {"color": THEME["text"], "size": 15}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": THEME["text_muted"]},
                        "bar": {"color": pred["risk_color"]},
                        "steps": [
                            {"range": [0, 30], "color": "#064E3B"},
                            {"range": [30, 60], "color": "#78350F"},
                            {"range": [60, 100], "color": "#7F1D1D"},
                        ],
                        "threshold": {
                            "line": {"color": "#FFFFFF", "width": 3},
                            "thickness": 0.75,
                            "value": 50,
                        },
                    },
                )
            )
            fig_gauge.update_layout(
                template="plotly_dark",
                paper_bgcolor=THEME["card"],
                plot_bgcolor=THEME["card"],
                height=280,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with res_col2:
            st.markdown("### Prediction Outcome")
            if pred["risk_category"] == "High Risk":
                st.markdown('<span class="risk-high">HIGH CHURN RISK</span>', unsafe_allow_html=True)
            elif pred["risk_category"] == "Medium Risk":
                st.markdown('<span class="risk-med">MEDIUM CHURN RISK</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="risk-low">LOW CHURN RISK</span>', unsafe_allow_html=True)

            st.markdown(f"**Predicted Action**: `{pred['churn_label']}`")
            st.markdown(f"**Calculated Risk Score**: `{prob*100:.2f}%`")
            st.markdown(f"**Decision Threshold**: `50.0%`")

            # Link to downstream modules
            st.info("👉 Switch to **Explainability Engine** or **Retention Engine** to analyze drivers and view recommended interventions.")


# ==========================================
# 2. CUSTOMER SEGMENTATION SECTION
# ==========================================
elif menu_option == "👥 2. Customer Segmentation":
    st.markdown('<div class="main-title">👥 Customer Segmentation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Unsupervised K-Means clustering grouping customers by tenure, monthly spend, lifetime value, and service adoption.</div>',
        unsafe_allow_html=True,
    )

    summary_df = segmenter.cluster_summary

    # Segment Cards
    cols = st.columns(len(summary_df))
    for idx, (col, row) in enumerate(zip(cols, summary_df.iterrows())):
        r = row[1]
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="font-weight:700; color:#FFFFFF; font-size:1.05rem; margin-bottom:4px;">{r['Segment_Name']}</div>
                    <div style="font-size:0.82rem; color:{THEME['text_muted']};">Cluster {int(r['Cluster'])} • {r['Percentage']:.1f}% share</div>
                    <div class="metric-val" style="margin-top:8px; font-size:1.45rem; color:{THEME['primary']};">{r['Count']:,}</div>
                    <div style="font-size:0.85rem; margin-top:6px; color:{THEME['text']}; line-height:1.6;">
                        Avg Tenure: <b style="color:#FFFFFF;">{r['Avg_Tenure']:.0f}m</b><br>
                        Monthly: <b style="color:#FFFFFF;">${r['Avg_MonthlyCharges']:.1f}</b><br>
                        Churn Rate: <b style="color:{THEME['churn'] if r['Churn_Rate'] > 0.3 else THEME['success']};">{r['Churn_Rate']*100:.1f}%</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Visualizations
    col_chart1, col_chart2 = st.columns([3, 2])
    with col_chart1:
        st.subheader("Customer Personas Scatter Plot (Tenure vs Monthly Charges)")
        clean_df_segmented = segmenter.fit(clean_df)
        fig_scatter = px.scatter(
            clean_df_segmented,
            x="tenure",
            y="MonthlyCharges",
            color="Segment_Name",
            size="TotalCharges",
            hover_data=["customerID", "Contract", "Churn"],
            labels={"tenure": "Tenure (Months)", "MonthlyCharges": "Monthly Charges ($)"},
            color_discrete_sequence=[THEME["primary"], THEME["secondary"], THEME["warning"], THEME["success"]],
            opacity=0.75,
        )
        style_plot(fig_scatter, height=420)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_chart2:
        st.subheader("Segment Share & Churn Risk")
        fig_donut = px.pie(
            summary_df,
            values="Count",
            names="Segment_Name",
            hole=0.48,
            color_discrete_sequence=[THEME["primary"], THEME["secondary"], THEME["warning"], THEME["success"]],
        )
        style_plot(fig_donut, height=420)
        st.plotly_chart(fig_donut, use_container_width=True)

    # Detailed Characteristics Table
    st.subheader("Segment Profiles & Tailored Business Strategies")
    display_summary = summary_df.copy()
    display_summary["Avg_Tenure"] = display_summary["Avg_Tenure"].map("{:.1f} months".format)
    display_summary["Avg_MonthlyCharges"] = display_summary["Avg_MonthlyCharges"].map("${:.2f}".format)
    display_summary["Avg_TotalCharges"] = display_summary["Avg_TotalCharges"].map("${:.2f}".format)
    display_summary["Churn_Rate"] = display_summary["Churn_Rate"].map("{:.1%}".format)
    display_summary["Percentage"] = display_summary["Percentage"].map("{:.1f}%".format)

    render_table(
        display_summary[
            [
                "Cluster",
                "Segment_Name",
                "Count",
                "Percentage",
                "Avg_Tenure",
                "Avg_MonthlyCharges",
                "Avg_TotalCharges",
                "Churn_Rate",
                "Risk_Level",
                "Strategy",
            ]
        ]
    )


# ==========================================
# 3. EXPLAINABILITY ENGINE (SHAP)
# ==========================================
elif menu_option == "🔍 3. Explainability Engine":
    st.markdown('<div class="main-title">🔍 Explainability Engine (SHAP)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">SHAP (SHapley Additive exPlanations) tree-based interpretability for the actual trained XGBoost model.</div>',
        unsafe_allow_html=True,
    )

    tab_global, tab_local = st.tabs(["🌐 Global Feature Importance", "🔬 Individual Customer Explanation"])

    with tab_global:
        st.subheader("Global SHAP Feature Importance")
        st.write(
            "Mean absolute SHAP value $|SHAP|$ measures how much each feature pushes the model output across the entire customer base."
        )
        top_shap = explainer.get_global_importance(top_n=12)
        fig_gshap = px.bar(
            top_shap.sort_values("mean_abs_shap", ascending=True),
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            color="mean_abs_shap",
            color_continuous_scale=[[0, "#1E293B"], [1, THEME["secondary"]]],
            labels={"mean_abs_shap": "Mean |SHAP Value|", "feature": "Feature"},
            text="percentage",
        )
        fig_gshap.update_traces(texttemplate="%{text:.1f}% impact", textposition="outside")
        fig_gshap.update_layout(showlegend=False)
        style_plot(fig_gshap, height=450)
        st.plotly_chart(fig_gshap, use_container_width=True)

        st.caption(
            "Key Observation: Contract type (Month-to-month) and customer tenure are the dominant factors determining retention across the company."
        )

    with tab_local:
        st.subheader("Individual Prediction Decomposition")
        col_c1, col_c2 = st.columns([1, 2])
        with col_c1:
            cust_id = st.selectbox(
                "Select Customer ID to Explain:",
                options=raw_df["customerID"].head(100).tolist(),
                index=0,
            )
            sample_data = raw_df[raw_df["customerID"] == cust_id].iloc[0].to_dict()
            encoded_sample = processor.transform_single_customer(sample_data)
            pred_res = predictor.predict_single_customer(sample_data, processor)

            st.markdown(
                f"""
                <div class="info-box-dark">
                    <span style="color:{THEME['text_muted']};">Customer ID:</span> <b style="color:#FFFFFF;">{cust_id}</b><br>
                    <span style="color:{THEME['text_muted']};">Predicted Probability:</span> <b style="color:{pred_res['risk_color']};">{pred_res['churn_probability']*100:.1f}%</b><br>
                    <span style="color:{THEME['text_muted']};">Assigned Risk:</span> <b style="color:#FFFFFF;">{pred_res['risk_category']}</b>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_c2:
            exp_res = explainer.explain_customer(encoded_sample, top_n=5)
            # Create Plotly Waterfall / Impact chart
            factors = exp_res["risk_drivers"] + exp_res["protective_factors"]
            factors_df = pd.DataFrame(factors).sort_values("shap_value", ascending=True)

            factors_df["color"] = factors_df["shap_value"].apply(
                lambda x: THEME["churn"] if x > 0 else THEME["success"]
            )
            factors_df["feature_label"] = factors_df["feature"].apply(explainer._format_feature_name)

            fig_waterfall = go.Figure(
                go.Bar(
                    x=factors_df["shap_value"],
                    y=factors_df["feature_label"],
                    orientation="h",
                    marker_color=factors_df["color"],
                    text=factors_df["shap_value"].apply(lambda x: f"{x:+.2f}"),
                    textposition="outside",
                )
            )
            style_plot(fig_waterfall, height=350, title="Top Contributors to Churn Risk (Red: Risk Driver | Green: Retention Anchor)")
            st.plotly_chart(fig_waterfall, use_container_width=True)

        st.markdown("### 💬 Plain Language Explanation")
        for sent in exp_res["human_explanations"]:
            st.markdown(f"• {sent}")


# ==========================================
# 4. RETENTION ENGINE
# ==========================================
elif menu_option == "💡 4. Retention Engine":
    st.markdown('<div class="main-title">💡 Retention Engine</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Automated, rule-based recommendation system that aligns customer risk drivers with targeted retention offers.</div>',
        unsafe_allow_html=True,
    )

    col_pick, col_info = st.columns([1, 2])
    with col_pick:
        target_id = st.selectbox(
            "Select Customer for Retention Plan:",
            options=raw_df["customerID"].head(100).tolist(),
            index=2,
        )
        target_cust = raw_df[raw_df["customerID"] == target_id].iloc[0].to_dict()
        pred_res = predictor.predict_single_customer(target_cust, processor)

    with col_info:
        st.markdown(
            f"""
            <div class="info-box-dark">
                <span style="color:{THEME['text_muted']};">Customer Profile:</span> <b style="color:#FFFFFF;">{target_id}</b> | 
                <span style="color:{THEME['text_muted']};">Tenure:</span> <b style="color:#FFFFFF;">{target_cust['tenure']}m</b> | 
                <span style="color:{THEME['text_muted']};">Monthly Bill:</span> <b style="color:#FFFFFF;">${target_cust['MonthlyCharges']}</b> | 
                <span style="color:{THEME['text_muted']};">Contract:</span> <b style="color:#FFFFFF;">{target_cust['Contract']}</b> | 
                <span style="color:{THEME['text_muted']};">Internet:</span> <b style="color:#FFFFFF;">{target_cust['InternetService']}</b> | 
                <span style="color:{THEME['text_muted']};">Payment:</span> <b style="color:#FFFFFF;">{target_cust['PaymentMethod']}</b><br>
                <div style="margin-top:6px;">
                    <span style="color:{THEME['text_muted']};">Churn Risk Level:</span> <span style="font-weight:700; color:{pred_res['risk_color']}">{pred_res['churn_probability']*100:.1f}% ({pred_res['risk_category']})</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Generate Recommendations
    rec_package = retention.generate_recommendations(target_cust, pred_res)

    st.markdown(
        f"""
        <div style="background:{rec_package['urgency_color']}20; border-left: 5px solid {rec_package['urgency_color']}; padding: 14px 18px; border-radius: 8px; margin-bottom: 22px;">
            <div style="font-weight:700; color:{rec_package['urgency_color']}; font-size:1.02rem; letter-spacing:0.5px;">{rec_package['urgency']}</div>
            <div style="color:#E2E8F0; margin-top:4px; font-size:0.95rem;">{rec_package['headline']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader(f"Recommended Interventions ({rec_package['total_recommendations']})")

    for rec in rec_package["recommendations"]:
        with st.container():
            st.markdown(
                f"""
                <div style="background:{THEME['card']}; border:1px solid {THEME['border']}; border-radius:10px; padding:18px; margin-bottom:14px; box-shadow:0 4px 6px -1px rgba(0,0,0,0.2);">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="background:rgba(59,130,246,0.15); color:#60A5FA; border:1px solid rgba(59,130,246,0.3); padding:4px 12px; border-radius:14px; font-size:0.75rem; font-weight:700; text-transform:uppercase;">{rec['category']}</span>
                        <span style="font-size:0.82rem; color:{THEME['text_muted']};">Cost: <b style="color:#FFFFFF;">{rec['financial_cost']}</b></span>
                    </div>
                    <div style="font-size:1.15rem; font-weight:700; color:#FFFFFF; margin-top:10px;">{rec['title']}</div>
                    <div style="color:#CBD5E1; font-size:0.95rem; margin-top:6px; line-height:1.5;"><b style="color:#FFFFFF;">Strategy:</b> {rec['action']}</div>
                    <div style="color:#34D399; font-size:0.88rem; font-weight:600; margin-top:8px;">✔ Expected Impact: {rec['expected_impact']}</div>
                    <div style="color:{THEME['text_muted']}; font-size:0.82rem; margin-top:4px;"><i>Reason: {rec['reason']}</i></div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==========================================
# 5. WHAT-IF SIMULATOR & REVENUE AT RISK
# ==========================================
elif menu_option == "⚡ 5. What-If Simulator & Revenue at Risk":
    st.markdown('<div class="main-title">⚡ What-If Simulator & Revenue at Risk</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Test hypothetical retention scenarios and evaluate overall company financial revenue exposure.</div>',
        unsafe_allow_html=True,
    )

    sub_tab_sim, sub_tab_rar = st.tabs(["🎛️ What-If Scenario Simulator", "💰 Portfolio Revenue at Risk"])

    with sub_tab_sim:
        st.subheader("Simulate Customer Changes & Churn Probability Delta")
        sim_id = st.selectbox(
            "Select Baseline Customer:",
            options=raw_df["customerID"].head(100).tolist(),
            index=0,
        )
        base_cust = raw_df[raw_df["customerID"] == sim_id].iloc[0].to_dict()

        st.caption("Modify customer attributes below to observe predicted changes in churn probability:")
        sim_col1, sim_col2, sim_col3 = st.columns(3)

        with sim_col1:
            new_contract = st.selectbox(
                "Modify Contract Type:",
                ["Month-to-month", "One year", "Two year"],
                index=["Month-to-month", "One year", "Two year"].index(base_cust.get("Contract", "Month-to-month")),
            )
        with sim_col2:
            new_monthly = st.slider(
                "Modify Monthly Charges ($):",
                18.0,
                120.0,
                float(base_cust.get("MonthlyCharges", 70.0)),
                step=1.0,
            )
        with sim_col3:
            new_tech_support = st.selectbox(
                "Modify Tech Support:",
                ["No", "Yes"],
                index=1 if base_cust.get("TechSupport") == "Yes" else 0,
            )

        modifications = {
            "Contract": new_contract,
            "MonthlyCharges": new_monthly,
            "TechSupport": new_tech_support,
        }

        # Run Simulator
        sim_result = simulator.simulate(base_cust, modifications)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(
                f'<div class="metric-card"><div class="metric-val">{sim_result["original_probability"]*100:.1f}%</div><div class="metric-lbl">Baseline Churn Risk</div></div>',
                unsafe_allow_html=True,
            )
        with m2:
            delta_val = sim_result["probability_difference"] * 100
            delta_color = THEME["success"] if delta_val < 0 else THEME["churn"]
            st.markdown(
                f'<div class="metric-card"><div class="metric-val" style="color:{delta_color};">{sim_result["modified_probability"]*100:.1f}% <span style="font-size:1rem;">({delta_val:+.1f}%)</span></div><div class="metric-lbl">Simulated Churn Risk</div></div>',
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f'<div class="metric-card"><div class="metric-val" style="color:#FFFFFF;">{sim_result["modified_risk_category"]}</div><div class="metric-lbl">Risk Category Shift</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Before / After Comparison Bar Chart
        comp_df = pd.DataFrame(
            {
                "Scenario": ["Original Baseline", "Hypothetical Scenario"],
                "Churn Probability (%)": [
                    sim_result["original_probability"] * 100,
                    sim_result["modified_probability"] * 100,
                ],
            }
        )
        fig_comp = px.bar(
            comp_df,
            x="Scenario",
            y="Churn Probability (%)",
            text="Churn Probability (%)",
            color="Scenario",
            color_discrete_sequence=["#475569", THEME["primary"]],
        )
        fig_comp.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_comp.update_layout(showlegend=False, yaxis_range=[0, 100])
        style_plot(fig_comp, height=300)
        st.plotly_chart(fig_comp, use_container_width=True)

        st.warning(f"⚠️ **Notice**: {sim_result['disclaimer']}")

    with sub_tab_rar:
        st.subheader("Company-Wide Portfolio Revenue Exposure")
        rar_res = revenue_analyzer.analyze(raw_df)
        kpis = rar_res["summary_kpis"]

        # KPI Row
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.markdown(
                f'<div class="metric-card"><div class="metric-val">{kpis["total_customers"]:,}</div><div class="metric-lbl">Total Customers</div></div>',
                unsafe_allow_html=True,
            )
        with r2:
            st.markdown(
                f'<div class="metric-card"><div class="metric-val" style="color:{THEME["primary"]};">${kpis["total_monthly_revenue"]:,.0f}</div><div class="metric-lbl">Total Monthly Revenue</div></div>',
                unsafe_allow_html=True,
            )
        with r3:
            st.markdown(
                f'<div class="metric-card"><div class="metric-val" style="color:{THEME["churn"]};">${kpis["expected_monthly_risk"]:,.0f}</div><div class="metric-lbl">Expected Monthly Risk</div></div>',
                unsafe_allow_html=True,
            )
        with r4:
            st.markdown(
                f'<div class="metric-card"><div class="metric-val" style="color:{THEME["churn"]};">${kpis["expected_annual_risk"]:,.0f}</div><div class="metric-lbl">Annualized Revenue at Risk</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Segment Revenue at Risk Breakdown
        col_s1, col_s2 = st.columns([1, 1])
        seg_rar_df = rar_res["segment_breakdown"]

        with col_s1:
            st.subheader("Expected Revenue at Risk by Segment")
            fig_rar_pie = px.pie(
                seg_rar_df,
                values="Expected_Monthly_Risk",
                names="Segment_Name",
                hole=0.48,
                color_discrete_sequence=[THEME["churn"], THEME["primary"], THEME["success"], THEME["secondary"]],
            )
            style_plot(fig_rar_pie, height=360)
            st.plotly_chart(fig_rar_pie, use_container_width=True)

        with col_s2:
            st.subheader("Segment Monthly Risk ($)")
            fig_rar_bar = px.bar(
                seg_rar_df,
                x="Segment_Name",
                y="Expected_Monthly_Risk",
                text="Expected_Monthly_Risk",
                color="Segment_Name",
                color_discrete_sequence=[THEME["churn"], THEME["primary"], THEME["success"], THEME["secondary"]],
            )
            fig_rar_bar.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
            fig_rar_bar.update_layout(showlegend=False)
            style_plot(fig_rar_bar, height=360)
            st.plotly_chart(fig_rar_bar, use_container_width=True)

        st.subheader("Segment Financial Breakdown")
        display_rar = seg_rar_df.copy()
        display_rar["Total_Monthly_Revenue"] = display_rar["Total_Monthly_Revenue"].map("${:,.2f}".format)
        display_rar["Expected_Monthly_Risk"] = display_rar["Expected_Monthly_Risk"].map("${:,.2f}".format)
        display_rar["Annual_Expected_Risk"] = display_rar["Annual_Expected_Risk"].map("${:,.2f}".format)
        display_rar["Avg_Churn_Probability"] = display_rar["Avg_Churn_Probability"].map("{:.1%}".format)
        display_rar["Risk_Share_Pct"] = display_rar["Risk_Share_Pct"].map("{:.1f}%".format)

        render_table(
            display_rar[
                [
                    "Segment_Name",
                    "Total_Customers",
                    "Total_Monthly_Revenue",
                    "Predicted_Churn_Count",
                    "Avg_Churn_Probability",
                    "Expected_Monthly_Risk",
                    "Annual_Expected_Risk",
                    "Risk_Share_Pct",
                ]
            ]
        )

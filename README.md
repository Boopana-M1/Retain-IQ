# Retain-IQ: Customer Churn Prediction & Retention Intelligence System

A clean, production-grade Customer Churn Prediction & Retention Intelligence System built using the original IBM Telco Customer Churn dataset.

Designed as an end-to-end Data Science project adhering to best practices in machine learning, model explainability (SHAP), customer segmentation (K-Means), and financial revenue risk modeling.

---

## 🌟 Key Features & Architecture

The system is organized into six core modules:

### Module 1: Data Processing & Feature Engineering
- **Missing Value Handling**: Handles 11 blank whitespace values in `TotalCharges` (for customers with zero tenure) by converting to numeric and replacing with `0.0`.
- **Type Conversions**: Standardizes categorical and numeric datatypes.
- **Engineered Features**:
  - `tenure_group`: Lifecycle cohorts (`0-12m`, `13-24m`, `25-48m`, `49-60m`, `60m+`).
  - `num_services`: Count of active subscription add-ons (Security, Backup, Streaming, Tech Support).
  - `charges_per_tenure`: Rate of bill accumulation.
  - Signal indicators: `is_month_to_month`, `is_fiber_optic`, `is_electronic_check`.
- **Data Leakage Prevention**: Encapsulates preprocessing in `TelcoDataProcessor` so training and single-customer inference share identical schemas.

### Module 2: Churn Prediction (XGBoost Classifier)
- **Algorithm**: `XGBClassifier` with histogram-based tree method.
- **Class Imbalance Compensation**: Utilizes `scale_pos_weight` ($2.77:1$) to avoid majority-class bias.
- **Stratified Splitting**: 80/20 stratified split preserving class distributions across train and test sets.
- **Evaluation Performance**:
  - **Accuracy**: `75.51%`
  - **ROC-AUC**: `84.57%`
  - **Recall**: `79.14%` *(prioritizes catching actual churners early)*
  - **Precision**: `52.58%`
  - **F1-Score**: `63.18%`

### Module 3: Customer Segmentation (K-Means Clustering)
- **Algorithm**: `KMeans` ($k=4$) with `StandardScaler` normalization.
- **Attributes**: `tenure`, `MonthlyCharges`, `TotalCharges`, `num_services`.
- **4 Distinct Personas**:
  1. **High-Value Loyalists**: High tenure (59.8m), high spend ($91.26/mo), multi-service users (14.4% churn).
  2. **New / Budget Month-to-Month**: Low tenure (8.8m), entry-level spend ($37.81/mo) (32.2% churn).
  3. **Heavy Streaming High-Spenders**: Mid-tenure (18.3m), high monthly bill ($79.78/mo) (42.7% churn).
  4. **Long-Term Budget Minimalists**: High tenure (53.3m), basic phone/DSL lines ($29.94/mo) (4.3% churn).

### Module 4: Explainability Engine (SHAP)
- **Global Interpretability**: Computes mean absolute SHAP values across portfolio features.
- **Local Explanation Decomposition**: Breaks down any customer's churn risk into:
  - **Risk Factors (+)**: Attributes pushing toward churn (e.g., Month-to-Month, Fiber Optic, Paperless Billing).
  - **Protective Factors (-)**: Attributes anchoring retention (e.g., Two-Year Contract, High Tenure, Tech Support).
- **Interactive Visuals**: Plotly horizontal impact charts and clear plain-language summaries.

### Module 5: Retention Engine
- **Intelligent Rule-Based Recommendations**: Aligns customer risk profiles with tailored business interventions:
  - *Month-to-month contract* → 1-Year or 2-Year contract incentive with 15% discount or bill credits.
  - *High monthly charges* → Personalized bundle restructuring or 6-month loyalty credits.
  - *Short tenure* → Proactive onboarding concierge and mobile app engagement.
  - *No tech support* → Complimentary 3-month priority 24/7 tech support trial.
  - *Electronic check* → One-time $10 bill credit to transition to automated payment.
- **Urgency Levels**: Dynamic urgency classification (Immediate Intervention, Proactive Engagement, Account Stable).

### Module 6: What-If Simulator & Revenue at Risk
- **What-If Scenario Simulator**:
  - Interactively alter contract type, monthly charges, or tech support to evaluate predicted delta in churn probability.
  - Side-by-side baseline vs scenario comparison.
  - Clear transparency disclaimer included.
- **Portfolio Revenue at Risk**:
  - Calculates total portfolio revenue ($456,116.60/month).
  - Binary predicted churn exposure ($211,464.60/month).
  - Probability-weighted expected revenue at risk ($203,950.81/month | $2,447,409.76/year).
  - Detailed segment-level financial exposure breakdown.

---

## 🛠️ Technology Stack

- **Language**: Python 3.11
- **Data & Computation**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn, XGBoost, Joblib
- **Model Explainability**: SHAP (TreeExplainer)
- **Interactive Visualizations**: Plotly Express & Plotly Graph Objects
- **Web Application**: Streamlit

---

## 📂 Project Organization

```text
Retain-IQ/
│
├── data/
│   └── telco_customer_churn.csv    # Original IBM dataset (7,043 rows, 21 columns)
│
├── models/
│   ├── data_processor.joblib        # Fitted preprocessing and feature schema
│   ├── churn_xgb_model.joblib       # Trained XGBoost Classifier model
│   ├── model_metrics.joblib         # Evaluation metrics and confusion matrix
│   ├── segmentation_kmeans.joblib   # Fitted K-Means model, scaler, and persona profiles
│   └── shap_explainer.joblib        # Fitted SHAP TreeExplainer and global summary
│
├── src/
│   ├── __init__.py
│   ├── data_processor.py            # Module 1: Preprocessing & feature engineering
│   ├── churn_model.py               # Module 2: XGBoost training and inference
│   ├── segmentation.py              # Module 3: K-Means clustering and personas
│   ├── explainability.py            # Module 4: SHAP global & local interpretability
│   ├── retention_engine.py          # Module 5: Actionable recommendation engine
│   └── simulator.py                 # Module 6: What-If simulation & Revenue at Risk
│
├── tests/
│   ├── test_data_processor.py       # Module 1 test suite
│   ├── test_retention_engine.py     # Module 5 test suite
│   └── test_simulator.py            # Module 6 test suite
│
├── train_churn_model.py             # Script to train Module 2 (XGBoost)
├── train_segmentation.py            # Script to train Module 3 (K-Means)
├── train_explainability.py          # Script to compute Module 4 (SHAP)
├── app.py                           # Complete Streamlit Dashboard Application
├── run_app.bat                      # One-click launcher for Windows
├── requirements.txt                 # Project dependencies
└── README.md                        # Documentation
```

---

## 🚀 Installation & Execution

### 1. Clone or Open Project Directory
```bash
cd Retain-IQ
```

### 2. Set Up Virtual Environment (Recommended)
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Retrain Models
All models are already pre-trained and saved in `models/`. To retrain:
```bash
python train_churn_model.py
python train_segmentation.py
python train_explainability.py
```

### 5. Run the Streamlit Application
```bash
streamlit run app.py
```
*(On Windows systems, you can also double-click `run_app.bat` or run `python -m streamlit run app.py`)*

The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## 📊 Summary of Model Performance

| Metric | Score | Explanation |
| :--- | :--- | :--- |
| **ROC-AUC** | **84.57%** | Strong discrimination ability between churners and non-churners |
| **Recall (Churn)** | **79.14%** | Captures ~80% of all potential churners in the test set |
| **Accuracy** | **75.51%** | Solid balanced accuracy across unbalanced classes |
| **F1-Score** | **63.18%** | Harmonized balance between precision and recall |
| **Precision** | **52.58%** | Optimized via class weights to prevent missing true churners |

---

## 📜 Dataset Reference
- **Source**: IBM Telco Customer Churn Dataset
- **URL**: [IBM ICP4D Telco Churn Repository](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)
- **Observations**: 7,043 customers, 21 columns

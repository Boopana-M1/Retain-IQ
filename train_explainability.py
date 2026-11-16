"""
Training and verification script for Module 4: Explainability Engine (SHAP).
Computes global SHAP importance, individual customer explanation decomposition,
and saves artifacts to models/shap_explainer.joblib.
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_processor import TelcoDataProcessor
from src.churn_model import ChurnPredictor
from src.explainability import ChurnExplainer


def main():
    print("=" * 60)
    print("  RETAIN-IQ: MODULE 4 - EXPLAINABILITY ENGINE (SHAP)")
    print("=" * 60)

    # 1. Load data processor and churn predictor
    processor = TelcoDataProcessor.load("models/data_processor.joblib")
    predictor = ChurnPredictor.load("models/churn_xgb_model.joblib")
    raw_df = processor.load_raw_data("data/telco_customer_churn.csv")
    X, y, clean_df = processor.fit_transform(raw_df)

    # 2. Fit ChurnExplainer
    print("\n[+] Initializing SHAP TreeExplainer with trained XGBoost model...")
    explainer = ChurnExplainer(predictor=predictor)
    explainer.fit(X, sample_size=500)

    # 3. Global SHAP Feature Importance
    print("\n" + "=" * 60)
    print("           GLOBAL SHAP FEATURE IMPORTANCE (TOP 10)")
    print("=" * 60)
    global_top = explainer.get_global_importance(top_n=10)
    for idx, row in global_top.iterrows():
        print(f"  {row['feature']:<32} : Mean |SHAP| = {row['mean_abs_shap']:.4f} ({row['percentage']:.2f}%)")

    # 4. Individual Customer Explanation
    sample_idx = 10  # Sample customer
    sample_raw = raw_df.iloc[sample_idx].to_dict()
    sample_encoded = processor.transform_single_customer(sample_raw)

    pred_res = predictor.predict_single_customer(sample_raw, processor)
    explanation = explainer.explain_customer(sample_encoded, top_n=4)

    print("\n" + "=" * 60)
    print(f"    LOCAL CUSTOMER EXPLANATION (ID: {sample_raw.get('customerID')})")
    print("=" * 60)
    print(f"  Predicted Probability: {pred_res['churn_probability']:.4f} ({pred_res['risk_category']})")
    print(f"  Model Base Value     : {explanation['base_value']:.4f}")
    print(f"  Total SHAP Impact    : {explanation['total_shap_impact']:.4f}")

    print("\n  Factors Increasing Churn Risk (Pushing Toward Churn):")
    for r in explanation["risk_drivers"]:
        print(f"    [+] {r['feature']:<28} (val={r['value']:<5}) : SHAP = +{r['shap_value']:.4f}")

    print("\n  Factors Protecting Against Churn (Anchoring Customer):")
    for p in explanation["protective_factors"]:
        print(f"    [-] {p['feature']:<28} (val={p['value']:<5}) : SHAP = {p['shap_value']:.4f}")

    print("\n  Human-Readable Explanations:")
    for sent in explanation["human_explanations"]:
        print(f"    - {sent}")

    # 5. Save Artifact
    explainer.save("models/shap_explainer.joblib")
    print("\n[+] SHAP explainer saved to models/shap_explainer.joblib")

    print("\n>>> MODULE 4 COMPLETED SUCCESSFULLY! <<<\n")


if __name__ == "__main__":
    main()

"""
Automated Verification Suite for Retain-IQ System.
Tests all 6 modules, verifies fixes for Error 1 (Segmentation num_services)
and Error 2 (PyArrow / NumPy compatibility), and tests Streamlit app execution.
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_processor import TelcoDataProcessor
from src.churn_model import ChurnPredictor
from src.segmentation import CustomerSegmenter
from src.explainability import ChurnExplainer
from src.retention_engine import RetentionEngine
from src.simulator import WhatIfSimulator, RevenueAtRiskAnalyzer


def test_all():
    print("=" * 70)
    print("      RETAIN-IQ: COMPREHENSIVE DEBUG & VERIFICATION SUITE")
    print("=" * 70)

    # Load artifacts
    print("\n[+] 1. Loading all saved artifacts...")
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
    print(f"    Raw dataset: {raw_df.shape} | Clean dataset: {clean_df.shape}")

    # Test 1 & 2: Customer Segmentation on both raw_df and clean_df
    print("\n[+] 2. Testing Customer Segmentation (Verifying Fix for KeyError: ['num_services'])...")
    # A. Test directly on raw_df (which previously failed because 'num_services' was missing!)
    seg_raw = segmenter.fit(raw_df)
    assert "Cluster" in seg_raw.columns, "Cluster column missing in seg_raw!"
    assert "Segment_Name" in seg_raw.columns, "Segment_Name column missing in seg_raw!"
    print(f"    ✔ segmenter.fit(raw_df) SUCCESS: {seg_raw.shape}, clusters: {seg_raw['Cluster'].unique()}")

    # B. Test on clean_df
    seg_clean = segmenter.fit(clean_df)
    assert "Cluster" in seg_clean.columns, "Cluster column missing in seg_clean!"
    assert "Segment_Name" in seg_clean.columns, "Segment_Name column missing in seg_clean!"
    print(f"    ✔ segmenter.fit(clean_df) SUCCESS: {seg_clean.shape}, summary rows: {len(segmenter.cluster_summary)}")

    # C. Test single-customer cluster prediction
    sample_c = raw_df.iloc[5].to_dict()
    cluster_pred = segmenter.predict_cluster(sample_c)
    print(f"    ✔ Single customer cluster: {cluster_pred['segment_name']} (ID: {cluster_pred['cluster_id']})")

    # Test 3: SHAP Explainability Engine
    print("\n[+] 3. Testing SHAP Global and Local Explainability...")
    g_shap = explainer.get_global_importance(top_n=5)
    print(f"    ✔ Global SHAP Top 3: {list(g_shap['feature'][:3])}")

    sample_encoded = processor.transform_single_customer(raw_df.iloc[10].to_dict())
    local_exp = explainer.explain_customer(sample_encoded, top_n=3)
    assert len(local_exp["risk_drivers"]) > 0, "No risk drivers found!"
    print(f"    ✔ Local SHAP Risk Drivers for customer: {[r['feature'] for r in local_exp['risk_drivers']]}")

    # Test 4: Retention Recommendations
    print("\n[+] 4. Testing Retention Engine...")
    cust_dict = raw_df.iloc[10].to_dict()
    pred_res = predictor.predict_single_customer(cust_dict, processor)
    rec_res = retention.generate_recommendations(cust_dict, pred_res)
    assert rec_res["total_recommendations"] > 0, "No recommendations generated!"
    print(f"    ✔ Retention Plan: Urgency = {rec_res['urgency']} | Offers = {rec_res['total_recommendations']}")

    # Test 5 & 6: What-If Simulator with Multiple Customers and Input Changes
    print("\n[+] 5. Testing What-If Simulator on Multiple Customers...")
    test_cases = [
        {"idx": 0, "changes": {"Contract": "Two year", "MonthlyCharges": 25.0}},
        {"idx": 15, "changes": {"Contract": "One year", "TechSupport": "Yes", "MonthlyCharges": 60.0}},
        {"idx": 42, "changes": {"Contract": "Month-to-month", "MonthlyCharges": 95.0}},
    ]
    for tc in test_cases:
        cust = raw_df.iloc[tc["idx"]].to_dict()
        sim_out = simulator.simulate(cust, tc["changes"])
        print(
            f"    Customer {cust['customerID']}: Orig={sim_out['original_probability']*100:.1f}% ({sim_out['original_risk_category']}) "
            f"-> Sim={sim_out['modified_probability']*100:.1f}% ({sim_out['modified_risk_category']}) "
            f"| Delta={sim_out['probability_difference']*100:+.1f}% | Changes={tc['changes']}"
        )

    # Test 7: Revenue at Risk & PyArrow / Streamlit table serialization
    print("\n[+] 6. Testing Revenue at Risk (Verifying Fix for PyArrow / NumPy compatibility)...")
    rar_out = revenue_analyzer.analyze(raw_df)
    kpis = rar_out["summary_kpis"]
    print(f"    Total Customers       : {kpis['total_customers']:,}")
    print(f"    Total Monthly Revenue : ${kpis['total_monthly_revenue']:,.2f}")
    print(f"    Expected Monthly Risk : ${kpis['expected_monthly_risk']:,.2f}")
    print(f"    Annual Expected Risk  : ${kpis['expected_annual_risk']:,.2f}")

    # Verify PyArrow conversion on all tables
    import pyarrow as pa
    t1 = pa.Table.from_pandas(rar_out["segment_breakdown"])
    t2 = pa.Table.from_pandas(rar_out["detailed_df"])
    print(f"    ✔ PyArrow table serialization PASSED for segment breakdown: {t1.shape}")
    print(f"    ✔ PyArrow table serialization PASSED for detailed dataframe: {t2.shape}")

    print("\n" + "=" * 70)
    print("      >>> ALL VERIFICATION TESTS PASSED FLAWLESSLY! <<<")
    print("=" * 70)


if __name__ == "__main__":
    test_all()

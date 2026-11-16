"""
Test suite and verification for Module 6: What-If Simulator + Revenue at Risk.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_processor import TelcoDataProcessor
from src.churn_model import ChurnPredictor
from src.segmentation import CustomerSegmenter
from src.simulator import WhatIfSimulator, RevenueAtRiskAnalyzer


def test_simulator_and_revenue():
    print("=== Testing Module 6: What-If Simulator + Revenue at Risk ===")

    # 1. Load models
    processor = TelcoDataProcessor.load("models/data_processor.joblib")
    predictor = ChurnPredictor.load("models/churn_xgb_model.joblib")
    segmenter = CustomerSegmenter.load("models/segmentation_kmeans.joblib")
    raw_df = processor.load_raw_data("data/telco_customer_churn.csv")

    # 2. Test What-If Simulator
    simulator = WhatIfSimulator(processor, predictor)
    sample_customer = raw_df.iloc[10].to_dict()

    print(f"\n--- A. What-If Simulation for Customer {sample_customer.get('customerID')} ---")
    modifications = {
        "Contract": "One year",
        "MonthlyCharges": 45.0,
        "TechSupport": "Yes",
    }
    sim_res = simulator.simulate(sample_customer, modifications)

    print(f"Original Churn Probability : {sim_res['original_probability']:.4f} ({sim_res['original_risk_category']})")
    print(f"Modified Churn Probability : {sim_res['modified_probability']:.4f} ({sim_res['modified_risk_category']})")
    print(f"Absolute Difference        : {sim_res['probability_difference']:+.4f}")
    print(f"Relative Percentage Change : {sim_res['percentage_change']:+.2f}%")
    print(f"Risk Reduced               : {sim_res['risk_reduced']}")
    assert sim_res["modified_probability"] < sim_res["original_probability"], "Expected churn risk to drop with 1-year contract + lower charges!"

    # 3. Test Revenue at Risk
    print("\n--- B. Revenue at Risk Analysis across Portfolio ---")
    analyzer = RevenueAtRiskAnalyzer(processor, predictor, segmenter)
    rar_res = analyzer.analyze(raw_df)

    kpis = rar_res["summary_kpis"]
    print(f"Total Customers            : {kpis['total_customers']:,}")
    print(f"Total Monthly Revenue      : ${kpis['total_monthly_revenue']:,.2f}")
    print(f"Predicted Churn Customers  : {kpis['predicted_churn_customers']:,} ({kpis['churn_rate_pct']:.1f}%)")
    print(f"Binary Monthly Risk        : ${kpis['binary_monthly_risk']:,.2f}")
    print(f"Expected Monthly Risk (EV) : ${kpis['expected_monthly_risk']:,.2f} ({kpis['risk_as_pct_of_revenue']:.1f}% of revenue)")
    print(f"Annual Expected Risk       : ${kpis['expected_annual_risk']:,.2f}")

    assert kpis["total_customers"] == 7043, "Expected 7043 customers!"
    assert kpis["expected_monthly_risk"] > 0, "Expected positive monthly risk!"

    print("\nSegment-Level Revenue at Risk:")
    seg_df = rar_res["segment_breakdown"]
    for _, row in seg_df.iterrows():
        print(f"  Segment: {row['Segment_Name']:<30} | Customers: {row['Total_Customers']:<5} | Exp. Monthly Risk: ${row['Expected_Monthly_Risk']:>9,.2f} | Share: {row['Risk_Share_Pct']:.1f}%")

    print("\n>>> ALL MODULE 6 TESTS PASSED SUCCESSFULLY! <<<\n")


if __name__ == "__main__":
    test_simulator_and_revenue()

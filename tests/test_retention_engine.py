"""
Unit test and verification for Module 5: Retention Engine.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retention_engine import RetentionEngine


def test_retention_engine():
    print("=== Testing Module 5: Retention Engine ===")
    engine = RetentionEngine()

    # Case 1: High risk customer
    high_risk_cust = {
        "MonthlyCharges": 85.5,
        "tenure": 3,
        "Contract": "Month-to-month",
        "TechSupport": "No",
        "OnlineSecurity": "No",
        "PaymentMethod": "Electronic check",
        "InternetService": "Fiber optic",
    }
    pred_high = {"churn_probability": 0.78, "risk_category": "High Risk"}
    rec_high = engine.generate_recommendations(high_risk_cust, pred_high)

    print(f"\n1. High Risk Customer (Risk: {pred_high['churn_probability']*100:.1f}%):")
    print(f"   Urgency: {rec_high['urgency']}")
    print(f"   Headline: {rec_high['headline']}")
    print(f"   Total recommendations: {rec_high['total_recommendations']}")
    assert rec_high["total_recommendations"] >= 4, "Expected multiple targeted recommendations!"
    for r in rec_high["recommendations"]:
        print(f"   - [{r['category']}] {r['title']}: {r['action'][:60]}...")

    # Case 2: Low risk loyal customer
    low_risk_cust = {
        "MonthlyCharges": 45.0,
        "tenure": 60,
        "Contract": "Two year",
        "TechSupport": "Yes",
        "OnlineSecurity": "Yes",
        "PaymentMethod": "Credit card (automatic)",
        "InternetService": "DSL",
    }
    pred_low = {"churn_probability": 0.08, "risk_category": "Low Risk"}
    rec_low = engine.generate_recommendations(low_risk_cust, pred_low)

    print(f"\n2. Low Risk Customer (Risk: {pred_low['churn_probability']*100:.1f}%):")
    print(f"   Urgency: {rec_low['urgency']}")
    print(f"   Headline: {rec_low['headline']}")
    assert rec_low["urgency"] == "ACCOUNT STABLE - NURTURE & ADVOCACY"
    print(f"   Recommendation: {rec_low['recommendations'][0]['title']}")

    print("\n>>> ALL MODULE 5 TESTS PASSED SUCCESSFULLY! <<<\n")


if __name__ == "__main__":
    test_retention_engine()

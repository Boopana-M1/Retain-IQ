"""
What-If Simulator and Revenue at Risk Module.
Provides interactive scenario analysis for retention strategies and
computes portfolio-wide and segment-level financial revenue at risk.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.data_processor import TelcoDataProcessor
from src.churn_model import ChurnPredictor
from src.segmentation import CustomerSegmenter


class WhatIfSimulator:
    """
    Simulation engine to assess how hypothetical service, contract,
    or pricing interventions affect customer churn probability.
    """

    DISCLAIMER = (
        "Hypothetical predictions are model-derived estimations based on historical patterns "
        "and do not guarantee real-world customer retention behavior."
    )

    def __init__(self, processor: TelcoDataProcessor, predictor: ChurnPredictor):
        self.processor = processor
        self.predictor = predictor

    def simulate(
        self,
        base_customer: Dict[str, Any],
        modifications: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Simulate the effect of attribute modifications on churn risk.
        """
        # Baseline customer prediction
        orig_encoded = self.processor.transform_single_customer(base_customer)
        orig_prob = float(self.predictor.predict_proba(orig_encoded)[0])

        # Apply modifications
        modified_customer = base_customer.copy()
        for k, v in modifications.items():
            modified_customer[k] = v

        mod_encoded = self.processor.transform_single_customer(modified_customer)
        mod_prob = float(self.predictor.predict_proba(mod_encoded)[0])

        diff = mod_prob - orig_prob
        pct_change = (diff / orig_prob * 100) if orig_prob > 0 else 0.0

        def get_risk_cat(prob: float) -> str:
            if prob < 0.30:
                return "Low Risk"
            elif prob < 0.60:
                return "Medium Risk"
            return "High Risk"

        orig_cat = get_risk_cat(orig_prob)
        mod_cat = get_risk_cat(mod_prob)

        return {
            "original_probability": orig_prob,
            "modified_probability": mod_prob,
            "probability_difference": diff,
            "percentage_change": pct_change,
            "original_risk_category": orig_cat,
            "modified_risk_category": mod_cat,
            "risk_reduced": diff < 0,
            "modifications_applied": modifications,
            "disclaimer": self.DISCLAIMER,
        }


class RevenueAtRiskAnalyzer:
    """
    Financial analytics engine to evaluate portfolio-wide and segment-level
    monthly and annual revenue exposure to customer churn.
    """

    def __init__(
        self,
        processor: TelcoDataProcessor,
        predictor: ChurnPredictor,
        segmenter: Optional[CustomerSegmenter] = None,
    ):
        self.processor = processor
        self.predictor = predictor
        self.segmenter = segmenter

    def analyze(self, raw_df: pd.DataFrame, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Compute comprehensive revenue at risk metrics across the dataset.
        """
        clean_df = self.processor.clean_data(raw_df)
        clean_df = self.processor.add_engineered_features(clean_df)

        X = self.processor.transform(clean_df)
        churn_probs = self.predictor.predict_proba(X)
        churn_preds = (churn_probs >= threshold).astype(int)

        clean_df["churn_probability"] = churn_probs
        clean_df["predicted_churn"] = churn_preds
        clean_df["expected_revenue_loss"] = clean_df["MonthlyCharges"] * clean_df["churn_probability"]

        # Assign segments if segmenter is provided
        if self.segmenter is not None:
            clean_df = self.segmenter.fit(clean_df)
        elif "Segment_Name" not in clean_df.columns:
            clean_df["Segment_Name"] = "All Customers"

        # Overall Portfolio Metrics
        total_customers = len(clean_df)
        total_monthly_revenue = float(clean_df["MonthlyCharges"].sum())
        total_predicted_churners = int(churn_preds.sum())
        churn_rate_pct = (total_predicted_churners / total_customers) * 100

        # Binary Churn Revenue at Risk
        binary_monthly_risk = float(clean_df[clean_df["predicted_churn"] == 1]["MonthlyCharges"].sum())
        binary_annual_risk = binary_monthly_risk * 12

        # Probability-Weighted Expected Revenue at Risk
        expected_monthly_risk = float(clean_df["expected_revenue_loss"].sum())
        expected_annual_risk = expected_monthly_risk * 12

        # Segment-Level Breakdown
        segment_df = (
            clean_df.groupby("Segment_Name")
            .agg(
                Total_Customers=("MonthlyCharges", "count"),
                Total_Monthly_Revenue=("MonthlyCharges", "sum"),
                Predicted_Churn_Count=("predicted_churn", "sum"),
                Expected_Monthly_Risk=("expected_revenue_loss", "sum"),
                Binary_Monthly_Risk=("MonthlyCharges", lambda s: s[clean_df.loc[s.index, "predicted_churn"] == 1].sum()),
                Avg_Churn_Probability=("churn_probability", "mean"),
            )
            .reset_index()
        )
        segment_df["Annual_Expected_Risk"] = segment_df["Expected_Monthly_Risk"] * 12
        segment_df["Risk_Share_Pct"] = (
            segment_df["Expected_Monthly_Risk"] / expected_monthly_risk * 100
        )
        segment_df["Segment_Churn_Rate"] = (
            segment_df["Predicted_Churn_Count"] / segment_df["Total_Customers"] * 100
        )

        return {
            "summary_kpis": {
                "total_customers": total_customers,
                "total_monthly_revenue": total_monthly_revenue,
                "total_annual_revenue": total_monthly_revenue * 12,
                "predicted_churn_customers": total_predicted_churners,
                "churn_rate_pct": churn_rate_pct,
                "binary_monthly_risk": binary_monthly_risk,
                "binary_annual_risk": binary_annual_risk,
                "expected_monthly_risk": expected_monthly_risk,
                "expected_annual_risk": expected_annual_risk,
                "risk_as_pct_of_revenue": (expected_monthly_risk / total_monthly_revenue) * 100,
            },
            "segment_breakdown": segment_df,
            "detailed_df": clean_df[
                [
                    "customerID",
                    "MonthlyCharges",
                    "TotalCharges",
                    "tenure",
                    "Contract",
                    "Segment_Name",
                    "churn_probability",
                    "predicted_churn",
                    "expected_revenue_loss",
                ]
            ] if "customerID" in clean_df.columns else clean_df,
        }

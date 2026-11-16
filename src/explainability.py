"""
Explainability Engine Module using SHAP (SHapley Additive exPlanations).
Provides global model explainability and local single-customer prediction decomposition
using the actual trained XGBoost model.
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import shap
import joblib

from src.data_processor import TelcoDataProcessor
from src.churn_model import ChurnPredictor


class ChurnExplainer:
    """
    SHAP-based explainability engine for XGBoost Churn Model.
    Deconstructs predictions into global importance and local risk/protective factors.
    """

    def __init__(self, predictor: Optional[ChurnPredictor] = None):
        self.predictor = predictor
        self.explainer: Optional[shap.TreeExplainer] = None
        self.feature_names: List[str] = []
        self.global_shap_summary: Optional[pd.DataFrame] = None
        self.background_sample: Optional[pd.DataFrame] = None
        self.background_shap_values: Optional[np.ndarray] = None
        self.base_value: float = 0.0

    def fit(self, X_train: pd.DataFrame, sample_size: int = 500) -> None:
        """
        Initialize TreeExplainer with trained XGBoost model and precompute background SHAP values.
        """
        if self.predictor is None or self.predictor.model is None:
            raise ValueError("Predictor with a trained XGBoost model is required.")

        self.feature_names = list(X_train.columns)
        self.explainer = shap.TreeExplainer(self.predictor.model)

        # Base value (log-odds expectation)
        self.base_value = float(self.explainer.expected_value)

        # Background sample for global analysis
        sample_n = min(sample_size, len(X_train))
        self.background_sample = X_train.sample(n=sample_n, random_state=42)

        # Compute SHAP values for background sample
        self.background_shap_values = self.explainer.shap_values(self.background_sample)

        # Compute mean absolute SHAP for global feature importance
        mean_abs_shap = np.abs(self.background_shap_values).mean(axis=0)
        self.global_shap_summary = pd.DataFrame(
            {
                "feature": self.feature_names,
                "mean_abs_shap": mean_abs_shap,
            }
        ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        self.global_shap_summary["percentage"] = (
            self.global_shap_summary["mean_abs_shap"]
            / self.global_shap_summary["mean_abs_shap"].sum()
        ) * 100

    def get_global_importance(self, top_n: int = 15) -> pd.DataFrame:
        """Return top N global features by mean absolute SHAP value."""
        if self.global_shap_summary is None:
            raise ValueError("Explainer not fitted. Call fit() or load() first.")
        return self.global_shap_summary.head(top_n)

    def explain_customer(
        self,
        customer_encoded: pd.DataFrame,
        top_n: int = 5,
    ) -> Dict[str, Any]:
        """
        Explain an individual customer's churn prediction.
        Breaks down factors that increase churn risk (+) vs reduce churn risk (-).
        """
        if self.explainer is None:
            raise ValueError("Explainer is not initialized. Call fit() or load() first.")

        # Compute SHAP values for single customer
        shap_vals = self.explainer.shap_values(customer_encoded)[0]

        contributions = []
        for feat, val, s_val in zip(
            self.feature_names, customer_encoded.iloc[0].values, shap_vals
        ):
            contributions.append(
                {
                    "feature": feat,
                    "value": float(val),
                    "shap_value": float(s_val),
                    "abs_shap": abs(float(s_val)),
                    "direction": "Increases Churn Risk" if s_val > 0 else "Reduces Churn Risk",
                }
            )

        contrib_df = pd.DataFrame(contributions).sort_values("abs_shap", ascending=False)

        # Top risk drivers (positive SHAP pushing towards churn)
        risk_drivers = (
            contrib_df[contrib_df["shap_value"] > 0]
            .sort_values("shap_value", ascending=False)
            .head(top_n)
            .to_dict(orient="records")
        )

        # Top retention anchors (negative SHAP reducing churn)
        protective_factors = (
            contrib_df[contrib_df["shap_value"] < 0]
            .sort_values("shap_value", ascending=True)
            .head(top_n)
            .to_dict(orient="records")
        )

        # Plain language summary sentences
        human_explanations = []
        for r in risk_drivers[:3]:
            human_explanations.append(
                f"Risk factor: '{self._format_feature_name(r['feature'])}' contributes strongly (+{r['shap_value']:.2f}) toward churn."
            )
        for p in protective_factors[:2]:
            human_explanations.append(
                f"Protective factor: '{self._format_feature_name(p['feature'])}' reduces churn probability ({p['shap_value']:.2f})."
            )

        return {
            "base_value": self.base_value,
            "total_shap_impact": float(shap_vals.sum()),
            "all_contributions": contrib_df.to_dict(orient="records"),
            "risk_drivers": risk_drivers,
            "protective_factors": protective_factors,
            "human_explanations": human_explanations,
        }

    def _format_feature_name(self, feature: str) -> str:
        """Format raw column names into readable labels."""
        friendly_names = {
            "is_month_to_month": "Month-to-Month Contract",
            "Contract_Two year": "Two-Year Contract",
            "Contract_One year": "One-Year Contract",
            "is_fiber_optic": "Fiber Optic Service",
            "InternetService_Fiber optic": "Fiber Optic Internet",
            "InternetService_No": "No Internet Service",
            "PaymentMethod_Electronic check": "Electronic Check Payment",
            "tenure": "Tenure (Months with Company)",
            "num_services": "Number of Subscribed Services",
            "MonthlyCharges": "Monthly Charges ($)",
            "TotalCharges": "Total Lifetime Charges ($)",
            "charges_per_tenure": "Charges Accumulation Rate",
            "TechSupport_Yes": "Active Tech Support",
            "OnlineSecurity_Yes": "Active Online Security",
            "OnlineBackup_Yes": "Active Online Backup",
            "PaperlessBilling": "Paperless Billing",
        }
        return friendly_names.get(feature, feature.replace("_", " "))

    def save(self, filepath: str = "models/shap_explainer.joblib") -> None:
        """Persist explainability artifacts."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(
            {
                "feature_names": self.feature_names,
                "base_value": self.base_value,
                "global_shap_summary": self.global_shap_summary,
                "background_sample": self.background_sample,
                "background_shap_values": self.background_shap_values,
            },
            filepath,
        )

    @classmethod
    def load(
        cls,
        filepath: str = "models/shap_explainer.joblib",
        predictor: Optional[ChurnPredictor] = None,
    ) -> "ChurnExplainer":
        """Load explainability artifacts from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"SHAP artifact not found at: {filepath}")

        payload = joblib.load(filepath)
        explainer_obj = cls(predictor=predictor)
        explainer_obj.feature_names = payload["feature_names"]
        explainer_obj.base_value = payload["base_value"]
        explainer_obj.global_shap_summary = payload["global_shap_summary"]
        explainer_obj.background_sample = payload.get("background_sample")
        explainer_obj.background_shap_values = payload.get("background_shap_values")

        if predictor is not None and predictor.model is not None:
            explainer_obj.explainer = shap.TreeExplainer(predictor.model)

        return explainer_obj

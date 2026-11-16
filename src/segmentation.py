"""
Customer Segmentation Module using K-Means Clustering.
Groups customers based on tenure, spend, and service usage patterns.
Provides cluster profiles, persona naming, and visual metrics.
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib


class CustomerSegmenter:
    """
    K-Means clustering engine for Telco customer persona segmentation.
    """

    SEGMENTATION_FEATURES = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "num_services",
    ]

    SERVICE_COLS = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans: Optional[KMeans] = None
        self.cluster_summary: Optional[pd.DataFrame] = None
        self.elbow_scores: Dict[str, List[float]] = {}
        self.personas: Dict[int, Dict[str, str]] = {}

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure all required segmentation features ('tenure', 'MonthlyCharges',
        'TotalCharges', 'num_services') exist and have valid numeric types.
        Safely derives 'num_services' and cleans 'TotalCharges' if called on raw data.
        """
        data = df.copy()

        # Clean TotalCharges if string / object with blanks
        if "TotalCharges" in data.columns:
            if data["TotalCharges"].dtype == object:
                data["TotalCharges"] = pd.to_numeric(
                    data["TotalCharges"].astype(str).str.strip(), errors="coerce"
                )
            data["TotalCharges"] = data["TotalCharges"].fillna(0.0)
        elif "MonthlyCharges" in data.columns and "tenure" in data.columns:
            data["TotalCharges"] = data["MonthlyCharges"] * data["tenure"]

        # Derive num_services if missing
        if "num_services" not in data.columns:
            active_count = pd.Series(0, index=data.index)
            for col in self.SERVICE_COLS:
                if col in data.columns:
                    active_count += (data[col] == "Yes").astype(int)
            data["num_services"] = active_count

        # Ensure numeric type for all segmentation features
        for col in self.SEGMENTATION_FEATURES:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0.0)
            else:
                data[col] = 0.0

        return data

    def _assign_dynamic_personas(self, data: pd.DataFrame) -> None:
        """Assign mathematically accurate persona definitions based on cluster centroids."""
        medians = data.groupby("Cluster")[["tenure", "MonthlyCharges"]].mean()
        self.personas = {}
        for c in range(self.n_clusters):
            avg_tenure = medians.loc[c, "tenure"] if c in medians.index else 0
            avg_monthly = medians.loc[c, "MonthlyCharges"] if c in medians.index else 0

            if avg_tenure >= 36 and avg_monthly >= 60:
                self.personas[c] = {
                    "name": "High-Value Loyalists",
                    "tagline": "Long-term subscribers with high spend and premium multi-service packages.",
                    "churn_risk": "Low",
                    "strategy": "VIP loyalty rewards, priority tech support, and early access to hardware upgrades.",
                }
            elif avg_tenure >= 36 and avg_monthly < 60:
                self.personas[c] = {
                    "name": "Long-Term Budget Minimalists",
                    "tagline": "Tenured customers with basic phone or DSL lines and consistent low spend.",
                    "churn_risk": "Low",
                    "strategy": "Retain via hassle-free auto-pay perks and gentle upselling to security add-ons.",
                }
            elif avg_tenure < 36 and avg_monthly >= 60:
                self.personas[c] = {
                    "name": "Heavy Streaming High-Spenders",
                    "tagline": "Recent to mid-tenure customers with high monthly bills sensitive to price hikes.",
                    "churn_risk": "High",
                    "strategy": "Value-pack streaming bundles, proactive bill review, and annual contract discount.",
                }
            else:
                self.personas[c] = {
                    "name": "New / Budget Month-to-Month",
                    "tagline": "Recent signups with low tenure and basic entry-level service usage.",
                    "churn_risk": "High",
                    "strategy": "Early onboarding check-ins, app engagement, and flexible contract incentives.",
                }

    def find_optimal_k(self, df: pd.DataFrame, max_k: int = 7) -> Dict[str, Any]:
        """Compute inertia and silhouette scores for k from 2 to max_k."""
        data = self._prepare_data(df)
        X = data[self.SEGMENTATION_FEATURES].copy()
        X_scaled = self.scaler.fit_transform(X)

        inertias = []
        silhouettes = []
        k_range = list(range(2, max_k + 1))

        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(X_scaled)
            inertias.append(float(km.inertia_))
            silhouettes.append(float(silhouette_score(X_scaled, labels)))

        self.elbow_scores = {
            "k_range": k_range,
            "inertias": inertias,
            "silhouettes": silhouettes,
        }
        return self.elbow_scores

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit K-Means on segmentation features and assign cluster labels.
        Returns the dataframe with 'Cluster' and 'Segment_Name' columns added.
        """
        data = self._prepare_data(df)
        X = data[self.SEGMENTATION_FEATURES]
        X_scaled = self.scaler.fit_transform(X)

        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        labels = self.kmeans.fit_predict(X_scaled)
        data["Cluster"] = labels
        self._assign_dynamic_personas(data)

        data["Segment_Name"] = data["Cluster"].map(
            lambda c: self.personas.get(c, {}).get("name", f"Segment {c}")
        )

        # Compute cluster characteristics
        self.cluster_summary = self._compute_cluster_summary(data)
        return data

    def _compute_cluster_summary(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate statistical profile for each customer cluster."""
        churn_series = data.get("Churn", None)
        if churn_series is not None:
            if churn_series.dtype == object:
                numeric_churn = (churn_series == "Yes").astype(int)
            else:
                numeric_churn = pd.to_numeric(churn_series, errors="coerce").fillna(0).astype(int)
        else:
            numeric_churn = pd.Series(0, index=data.index)

        data_with_churn = data.copy()
        data_with_churn["_numeric_churn"] = numeric_churn

        summary = (
            data_with_churn.groupby("Cluster")
            .agg(
                Count=("Cluster", "count"),
                Avg_Tenure=("tenure", "mean"),
                Avg_MonthlyCharges=("MonthlyCharges", "mean"),
                Avg_TotalCharges=("TotalCharges", "mean"),
                Avg_Services=("num_services", "mean"),
                Churn_Rate=("_numeric_churn", "mean"),
            )
            .reset_index()
        )
        summary["Percentage"] = (summary["Count"] / len(data)) * 100
        summary["Segment_Name"] = summary["Cluster"].map(
            lambda c: self.personas.get(c, {}).get("name", f"Segment {c}")
        )
        summary["Risk_Level"] = summary["Cluster"].map(
            lambda c: self.personas.get(c, {}).get("churn_risk", "Moderate")
        )
        summary["Strategy"] = summary["Cluster"].map(
            lambda c: self.personas.get(c, {}).get("strategy", "Standard Retention")
        )

        return summary

    def predict_cluster(self, customer_features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict the segment for a single customer record."""
        if self.kmeans is None:
            raise ValueError("KMeans model not fitted. Call fit() or load() first.")

        # Extract features safely
        tenure = float(customer_features.get("tenure", 0.0) or 0.0)
        monthly = float(customer_features.get("MonthlyCharges", 0.0) or 0.0)

        raw_tc = customer_features.get("TotalCharges", 0.0)
        if isinstance(raw_tc, str):
            try:
                total_charges = float(raw_tc.strip()) if raw_tc.strip() else monthly * tenure
            except ValueError:
                total_charges = monthly * tenure
        else:
            total_charges = float(raw_tc or 0.0)

        if "num_services" in customer_features and customer_features["num_services"] is not None:
            num_services = float(customer_features["num_services"])
        else:
            num_services = sum(1 for col in self.SERVICE_COLS if customer_features.get(col) == "Yes")

        X_input = pd.DataFrame(
            [[tenure, monthly, total_charges, num_services]],
            columns=self.SEGMENTATION_FEATURES,
        )
        scaled_vals = self.scaler.transform(X_input)
        cluster_id = int(self.kmeans.predict(scaled_vals)[0])

        persona = self.personas.get(
            cluster_id,
            {
                "name": f"Segment {cluster_id}",
                "tagline": "Custom customer segment.",
                "churn_risk": "Moderate",
                "strategy": "Monitor usage patterns.",
            },
        )
        return {
            "cluster_id": cluster_id,
            "segment_name": persona["name"],
            "tagline": persona["tagline"],
            "churn_risk": persona["churn_risk"],
            "recommended_strategy": persona["strategy"],
        }

    def save(self, filepath: str = "models/segmentation_kmeans.joblib") -> None:
        """Persist segmentation artifacts."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(
            {
                "n_clusters": self.n_clusters,
                "scaler": self.scaler,
                "kmeans": self.kmeans,
                "cluster_summary": self.cluster_summary,
                "elbow_scores": self.elbow_scores,
                "personas": self.personas,
            },
            filepath,
        )

    @classmethod
    def load(cls, filepath: str = "models/segmentation_kmeans.joblib") -> "CustomerSegmenter":
        """Load persisted segmentation artifacts."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Segmentation artifact not found at: {filepath}")

        payload = joblib.load(filepath)
        segmenter = cls(n_clusters=payload["n_clusters"])
        segmenter.scaler = payload["scaler"]
        segmenter.kmeans = payload["kmeans"]
        segmenter.cluster_summary = payload["cluster_summary"]
        segmenter.elbow_scores = payload.get("elbow_scores", {})
        segmenter.personas = payload.get("personas", {})
        return segmenter

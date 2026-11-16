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

from src.data_processor import TelcoDataProcessor


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

    CLUSTER_PERSONAS = {
        0: {
            "name": "New / Budget Month-to-Month",
            "tagline": "Recent signups with low tenure and basic service usage.",
            "churn_risk": "High",
            "strategy": "Early onboarding check-ins, app engagement, and flexible contract incentives.",
        },
        1: {
            "name": "High-Value Loyalists",
            "tagline": "Long-term subscribers with high monthly spend and premium multi-service packages.",
            "churn_risk": "Low",
            "strategy": "VIP loyalty rewards, priority tech support, and early access to hardware upgrades.",
        },
        2: {
            "name": "Heavy Streaming High-Spenders",
            "tagline": "Mid-tenure customers with high monthly fiber/streaming bills sensitive to price hikes.",
            "churn_risk": "Moderate-High",
            "strategy": "Value-pack streaming bundles, proactive bill review, and annual contract discount.",
        },
        3: {
            "name": "Long-Term Budget Minimalists",
            "tagline": "Tenured customers with basic phone or DSL lines and consistent low monthly spend.",
            "churn_risk": "Low",
            "strategy": "Retain via hassle-free auto-pay perks and gentle upselling to basic security add-ons.",
        },
    }

    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans: Optional[KMeans] = None
        self.cluster_summary: Optional[pd.DataFrame] = None
        self.elbow_scores: Dict[str, List[float]] = {}
        self.personas: Dict[int, Dict[str, str]] = {}

    def _assign_dynamic_personas(self, data: pd.DataFrame) -> None:
        """Assign mathematically accurate persona definitions based on cluster centroids."""
        medians = data.groupby("Cluster")[["tenure", "MonthlyCharges"]].mean()
        self.personas = {}
        for c in range(self.n_clusters):
            avg_tenure = medians.loc[c, "tenure"]
            avg_monthly = medians.loc[c, "MonthlyCharges"]

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
        X = df[self.SEGMENTATION_FEATURES].copy()
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
        data = df.copy()
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
        summary = (
            data.groupby("Cluster")
            .agg(
                Count=("Cluster", "count"),
                Avg_Tenure=("tenure", "mean"),
                Avg_MonthlyCharges=("MonthlyCharges", "mean"),
                Avg_TotalCharges=("TotalCharges", "mean"),
                Avg_Services=("num_services", "mean"),
                Churn_Rate=("Churn", lambda s: (s == 1).mean() if "Churn" in data.columns else 0.0),
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

        # Extract features as DataFrame with feature names to avoid sklearn warning
        feature_dict = {col: [float(customer_features.get(col, 0.0))] for col in self.SEGMENTATION_FEATURES}
        X_input = pd.DataFrame(feature_dict)
        scaled_vals = self.scaler.transform(X_input)
        cluster_id = int(self.kmeans.predict(scaled_vals)[0])

        # Persona lookup from fitted personas or default
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

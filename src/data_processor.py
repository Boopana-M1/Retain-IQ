"""
Data Processing and Feature Engineering Module for Telco Customer Churn.
Handles data loading, cleaning, type conversions, feature engineering,
and consistent train/prediction encoding to avoid data leakage.
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


class TelcoDataProcessor:
    """
    Robust data preprocessing pipeline for the IBM Telco Customer Churn dataset.
    Ensures identical feature shapes and encodings between training and inference.
    """

    BINARY_COLS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    SERVICE_COLS = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    CATEGORICAL_COLS = [
        "gender",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod",
        "tenure_group",
    ]
    NUMERIC_COLS = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "num_services",
        "charges_per_tenure",
        "is_month_to_month",
        "is_fiber_optic",
        "is_electronic_check",
    ]

    def __init__(self):
        self.feature_columns: List[str] = []
        self.scaler: Optional[StandardScaler] = None
        self.fitted: bool = False

    def load_raw_data(self, filepath: str = "data/telco_customer_churn.csv") -> pd.DataFrame:
        """Load the raw Telco dataset from CSV."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found at: {filepath}")
        df = pd.read_csv(filepath)
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data types and handle missing / blank values.
        Specifically handles blank whitespace strings in TotalCharges.
        """
        data = df.copy()

        # Handle blank spaces in TotalCharges
        if "TotalCharges" in data.columns:
            data["TotalCharges"] = pd.to_numeric(
                data["TotalCharges"].astype(str).str.strip(), errors="coerce"
            )
            # For customers with tenure == 0, TotalCharges is legitimately 0
            data["TotalCharges"] = data["TotalCharges"].fillna(0.0)

        # Standardize SeniorCitizen to int
        if "SeniorCitizen" in data.columns:
            data["SeniorCitizen"] = data["SeniorCitizen"].astype(int)

        # Clean binary target if present
        if "Churn" in data.columns and data["Churn"].dtype == object:
            data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})

        return data

    def add_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform domain-specific feature engineering:
        - tenure_group: customer lifecycle cohorts
        - num_services: count of active digital add-ons
        - charges_per_tenure: average monthly accumulation rate
        - is_month_to_month: high-risk contract indicator
        - is_fiber_optic: premium internet indicator
        - is_electronic_check: high churn payment method indicator
        """
        data = df.copy()

        # Tenure cohorts
        bins = [-1, 12, 24, 48, 60, 100]
        labels = ["0-12m", "13-24m", "25-48m", "49-60m", "60m+"]
        data["tenure_group"] = pd.cut(data["tenure"], bins=bins, labels=labels).astype(str)

        # Count of active subscription services
        active_services = pd.Series(0, index=data.index)
        for col in self.SERVICE_COLS:
            if col in data.columns:
                active_services += (data[col] == "Yes").astype(int)
        data["num_services"] = active_services

        # Effective charges ratio
        data["charges_per_tenure"] = data["TotalCharges"] / (data["tenure"] + 1)

        # High-signal risk indicators
        data["is_month_to_month"] = (data.get("Contract", "") == "Month-to-month").astype(int)
        data["is_fiber_optic"] = (data.get("InternetService", "") == "Fiber optic").astype(int)
        data["is_electronic_check"] = (data.get("PaymentMethod", "") == "Electronic check").astype(int)

        return data

    def fit_transform(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Optional[pd.Series], pd.DataFrame]:
        """
        Fit preprocessing encoders on training data and return (X, y, clean_df).
        """
        clean_df = self.clean_data(df)
        clean_df = self.add_engineered_features(clean_df)

        # Separate target if present
        y = clean_df["Churn"] if "Churn" in clean_df.columns else None

        # Build feature representation
        X = self._encode_features(clean_df, is_training=True)
        self.fitted = True

        return X, y, clean_df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new customer records using fitted schema to prevent schema mismatch.
        """
        if not self.fitted:
            raise ValueError("Processor must be fitted before calling transform().")

        clean_df = self.clean_data(df)
        clean_df = self.add_engineered_features(clean_df)
        X = self._encode_features(clean_df, is_training=False)
        return X

    def transform_single_customer(self, customer: Dict[str, Any]) -> pd.DataFrame:
        """Transform a single customer dictionary into model-ready features."""
        df = pd.DataFrame([customer])
        return self.transform(df)

    def _encode_features(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """Internal method to encode features consistently."""
        data = df.copy()

        # Binary mapping
        for col in self.BINARY_COLS:
            if col in data.columns:
                data[col] = data[col].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

        # Gender mapping
        if "gender" in data.columns:
            data["gender_Male"] = (data["gender"] == "Male").astype(int)

        # One-hot encode remaining multi-class categoricals
        cat_to_encode = [
            c for c in self.CATEGORICAL_COLS if c in data.columns and c not in self.BINARY_COLS and c != "gender"
        ]
        dummies = pd.get_dummies(data[cat_to_encode], drop_first=True, dtype=int)

        # Combine numeric and encoded categoricals
        feature_parts = []
        for col in self.BINARY_COLS:
            if col in data.columns:
                feature_parts.append(data[[col]])

        if "gender_Male" in data.columns:
            feature_parts.append(data[["gender_Male"]])

        for col in self.NUMERIC_COLS:
            if col in data.columns:
                feature_parts.append(data[[col]])

        feature_parts.append(dummies)
        encoded_df = pd.concat(feature_parts, axis=1)

        if is_training:
            self.feature_columns = list(encoded_df.columns)
            return encoded_df
        else:
            # Reindex to match exact training columns, filling missing with 0
            aligned_df = encoded_df.reindex(columns=self.feature_columns, fill_value=0)
            return aligned_df

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Perform stratified train-test split to preserve churn class ratio."""
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

    def save(self, filepath: str = "models/data_processor.joblib") -> None:
        """Save fitted processor to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str = "models/data_processor.joblib") -> "TelcoDataProcessor":
        """Load fitted processor from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Saved processor not found at: {filepath}")
        return joblib.load(filepath)

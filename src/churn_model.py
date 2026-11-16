"""
Churn Prediction Module using XGBoost Classifier.
Handles model training with class imbalance compensation,
evaluation metrics, feature importance, and inference.
"""

import os
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier
import joblib

from src.data_processor import TelcoDataProcessor


class ChurnPredictor:
    """
    XGBoost-based Churn Prediction Model with class imbalance handling.
    """

    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        self.model_params = model_params or {}
        self.model: Optional[XGBClassifier] = None
        self.feature_names: list = []
        self.metrics: Dict[str, Any] = {}
        self.feature_importance_df: Optional[pd.DataFrame] = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Train XGBoost Classifier with appropriate class imbalance handling (scale_pos_weight).
        """
        self.feature_names = list(X_train.columns)

        # Calculate class imbalance ratio: count(negative) / count(positive)
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / max(pos_count, 1)

        default_params = {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "logloss",
            "random_state": 42,
            "tree_method": "hist",
        }
        # Merge user params if provided
        final_params = {**default_params, **self.model_params}

        self.model = XGBClassifier(**final_params)

        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )

        # Calculate feature importances
        importances = self.model.feature_importances_
        fi_df = pd.DataFrame(
            {"feature": self.feature_names, "importance": importances}
        ).sort_values("importance", ascending=False).reset_index(drop=True)
        fi_df["percentage"] = (fi_df["importance"] / fi_df["importance"].sum()) * 100
        self.feature_importance_df = fi_df

        return self.get_feature_importance()

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Evaluate model performance on unseen test data.
        Returns accuracy, precision, recall, F1, ROC-AUC, and confusion matrix.
        """
        if self.model is None:
            raise ValueError("Model is not trained. Call train() or load() first.")

        y_prob = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        self.metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "matrix": cm.tolist(),
            },
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "threshold": threshold,
        }
        return self.metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return churn probability [0, 1] for input data."""
        if self.model is None:
            raise ValueError("Model is not trained.")
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Return binary prediction (0: No Churn, 1: Churn) based on threshold."""
        prob = self.predict_proba(X)
        return (prob >= threshold).astype(int)

    def predict_single_customer(
        self,
        customer_dict: Dict[str, Any],
        processor: TelcoDataProcessor,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Perform end-to-end churn prediction on a single customer record dictionary.
        Returns probability, binary prediction, risk category, and key metrics.
        """
        X_encoded = processor.transform_single_customer(customer_dict)
        churn_prob = float(self.predict_proba(X_encoded)[0])
        prediction = int(churn_prob >= threshold)

        # Risk categories: Low (<30%), Medium (30-60%), High (>60%)
        if churn_prob < 0.30:
            risk_category = "Low Risk"
            risk_color = "#28a745"
        elif churn_prob < 0.60:
            risk_category = "Medium Risk"
            risk_color = "#ffc107"
        else:
            risk_category = "High Risk"
            risk_color = "#dc3545"

        return {
            "churn_probability": churn_prob,
            "churn_prediction": prediction,
            "churn_label": "Likely to Churn" if prediction == 1 else "Likely to Stay",
            "risk_category": risk_category,
            "risk_color": risk_color,
            "threshold_used": threshold,
        }

    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """Return top N most important features as a DataFrame."""
        if self.feature_importance_df is None:
            raise ValueError("Model must be trained to extract feature importance.")
        return self.feature_importance_df.head(top_n)

    def save(
        self,
        model_path: str = "models/churn_xgb_model.joblib",
        metrics_path: str = "models/model_metrics.joblib",
    ) -> None:
        """Save model and evaluation artifacts to disk."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "feature_names": self.feature_names,
                "feature_importance_df": self.feature_importance_df,
                "model_params": self.model_params,
            },
            model_path,
        )
        if self.metrics:
            joblib.dump(self.metrics, metrics_path)

    @classmethod
    def load(
        cls,
        model_path: str = "models/churn_xgb_model.joblib",
        metrics_path: str = "models/model_metrics.joblib",
    ) -> "ChurnPredictor":
        """Load trained model and evaluation artifacts from disk."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        payload = joblib.load(model_path)
        predictor = cls(model_params=payload.get("model_params", {}))
        predictor.model = payload["model"]
        predictor.feature_names = payload.get("feature_names", [])
        predictor.feature_importance_df = payload.get("feature_importance_df", None)

        if os.path.exists(metrics_path):
            predictor.metrics = joblib.load(metrics_path)

        return predictor

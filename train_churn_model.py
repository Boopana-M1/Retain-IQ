"""
Training and Evaluation script for Module 2: Churn Prediction.
Loads data, performs stratified split, trains XGBoost, evaluates metrics,
prints confusion matrix and feature importance, and saves model artifacts.
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_processor import TelcoDataProcessor
from src.churn_model import ChurnPredictor


def main():
    print("=" * 60)
    print("  RETAIN-IQ: MODULE 2 - CHURN PREDICTION (XGBOOST)")
    print("=" * 60)

    # 1. Process Data
    processor = TelcoDataProcessor()
    raw_df = processor.load_raw_data("data/telco_customer_churn.csv")
    X, y, clean_df = processor.fit_transform(raw_df)

    processor.save("models/data_processor.joblib")
    print(f"[+] Data loaded and preprocessed. Total samples: {len(X)}, Features: {X.shape[1]}")

    # 2. Stratified Train-Test Split
    X_train, X_test, y_train, y_test = processor.split_data(X, y, test_size=0.2, random_state=42)
    print(f"[+] Stratified Split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
    print(f"    Train Churn Rate: {y_train.mean():.4f} | Test Churn Rate: {y_test.mean():.4f}")

    # 3. Train Model
    predictor = ChurnPredictor()
    print("\n[+] Training XGBoost Classifier with class-imbalance compensation...")
    predictor.train(X_train, y_train, X_val=X_test, y_val=y_test)

    # 4. Evaluate Performance
    metrics = predictor.evaluate(X_test, y_test, threshold=0.5)
    print("\n" + "=" * 40)
    print("        MODEL EVALUATION RESULTS")
    print("=" * 40)
    print(f"  Accuracy : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  Recall   : {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  F1-Score : {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    print(f"  ROC-AUC  : {metrics['roc_auc']:.4f} ({metrics['roc_auc']*100:.2f}%)")

    cm = metrics["confusion_matrix"]
    print("\nConfusion Matrix:")
    print(f"  True Negatives  (TN): {cm['tn']}")
    print(f"  False Positives (FP): {cm['fp']}")
    print(f"  False Negatives (FN): {cm['fn']}")
    print(f"  True Positives  (TP): {cm['tp']}")

    # 5. Top Feature Importance
    top_fi = predictor.get_feature_importance(top_n=10)
    print("\nTop 10 Feature Importances:")
    for idx, row in top_fi.iterrows():
        print(f"  {row['feature']:<30} : {row['importance']:.4f} ({row['percentage']:.2f}%)")

    # 6. Save Model
    predictor.save("models/churn_xgb_model.joblib", "models/model_metrics.joblib")
    print("\n[+] Model saved to models/churn_xgb_model.joblib")
    print("[+] Metrics saved to models/model_metrics.joblib")

    # 7. Verification: Predict on sample customer
    sample_raw = raw_df.iloc[10].to_dict()
    prediction_result = predictor.predict_single_customer(sample_raw, processor)
    print("\nSample Customer Prediction (Customer ID: {}):".format(sample_raw.get("customerID")))
    print(f"  Probability   : {prediction_result['churn_probability']:.4f}")
    print(f"  Classification: {prediction_result['churn_label']}")
    print(f"  Risk Category : {prediction_result['risk_category']}")

    print("\n>>> MODULE 2 COMPLETED SUCCESSFULLY! <<<\n")


if __name__ == "__main__":
    main()

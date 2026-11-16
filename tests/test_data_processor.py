"""
Test suite and verification for Module 1: Data Processing & Feature Engineering.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_processor import TelcoDataProcessor


def test_data_processor():
    print("=== Testing Module 1: Data Processing & Feature Engineering ===")
    processor = TelcoDataProcessor()

    # 1. Load data
    raw_df = processor.load_raw_data("data/telco_customer_churn.csv")
    print(f"1. Raw data loaded: {raw_df.shape}")
    assert raw_df.shape == (7043, 21), f"Expected (7043, 21), got {raw_df.shape}"

    # 2. Clean data
    clean_df = processor.clean_data(raw_df)
    assert clean_df["TotalCharges"].isnull().sum() == 0, "Missing values remain in TotalCharges!"
    assert clean_df["TotalCharges"].dtype in [np.float64, float], "TotalCharges is not float!"
    print(f"2. TotalCharges cleaned. Min: {clean_df['TotalCharges'].min()}, Max: {clean_df['TotalCharges'].max()}")

    # 3. Fit transform
    X, y, df_processed = processor.fit_transform(raw_df)
    print(f"3. Feature matrix X shape: {X.shape}, Target y shape: {y.shape}")
    assert X.shape[0] == 7043, f"Expected 7043 rows, got {X.shape[0]}"
    assert not X.isnull().values.any(), "Feature matrix contains NaN values!"
    assert y is not None, "Target y is None!"
    assert set(y.unique()) == {0, 1}, f"Unexpected target classes: {y.unique()}"

    # Check engineered features
    assert "num_services" in X.columns, "num_services missing!"
    assert "charges_per_tenure" in X.columns, "charges_per_tenure missing!"
    assert "is_month_to_month" in X.columns, "is_month_to_month missing!"
    print(f"Engineered features present: num_services (mean: {X['num_services'].mean():.2f}), charges_per_tenure (mean: {X['charges_per_tenure'].mean():.2f})")

    # 4. Stratified Split
    X_train, X_test, y_train, y_test = processor.split_data(X, y, test_size=0.2, random_state=42)
    print(f"4. Split completed: X_train={X_train.shape}, X_test={X_test.shape}")
    train_churn_rate = y_train.mean()
    test_churn_rate = y_test.mean()
    print(f"   Train churn rate: {train_churn_rate:.4f}, Test churn rate: {test_churn_rate:.4f}")
    assert abs(train_churn_rate - test_churn_rate) < 0.005, "Stratification failed!"

    # 5. Persistence (Save & Load)
    os.makedirs("models", exist_ok=True)
    processor.save("models/data_processor.joblib")
    loaded_processor = TelcoDataProcessor.load("models/data_processor.joblib")
    assert loaded_processor.feature_columns == processor.feature_columns, "Loaded processor columns do not match!"
    print(f"5. Saved and loaded processor successfully. Number of features: {len(loaded_processor.feature_columns)}")

    # 6. Single customer transform
    sample_customer = raw_df.iloc[0].to_dict()
    del sample_customer["customerID"]
    if "Churn" in sample_customer:
        del sample_customer["Churn"]

    sample_encoded = loaded_processor.transform_single_customer(sample_customer)
    print(f"6. Single customer transformed shape: {sample_encoded.shape}")
    assert sample_encoded.shape == (1, X.shape[1]), f"Expected (1, {X.shape[1]}), got {sample_encoded.shape}"
    assert list(sample_encoded.columns) == list(X.columns), "Column alignment mismatch!"

    print("\n>>> ALL MODULE 1 TESTS PASSED SUCCESSFULLY! <<<\n")


if __name__ == "__main__":
    test_data_processor()

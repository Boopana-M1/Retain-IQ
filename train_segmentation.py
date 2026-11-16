"""
Training script for Module 3: Customer Segmentation (K-Means Clustering).
Computes optimal k, fits K-Means, computes cluster characteristics,
and saves model artifacts.
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_processor import TelcoDataProcessor
from src.segmentation import CustomerSegmenter


def main():
    print("=" * 60)
    print("  RETAIN-IQ: MODULE 3 - CUSTOMER SEGMENTATION (K-MEANS)")
    print("=" * 60)

    # 1. Process data
    processor = TelcoDataProcessor()
    raw_df = processor.load_raw_data("data/telco_customer_churn.csv")
    _, _, clean_df = processor.fit_transform(raw_df)

    # 2. Find optimal k
    segmenter = CustomerSegmenter(n_clusters=4, random_state=42)
    print("\n[+] Evaluating cluster count k in range [2, 6]...")
    elbow_res = segmenter.find_optimal_k(clean_df, max_k=6)
    for k, inertia, sil in zip(
        elbow_res["k_range"], elbow_res["inertias"], elbow_res["silhouettes"]
    ):
        print(f"    k={k}: Inertia = {inertia:10.2f} | Silhouette Score = {sil:.4f}")

    # 3. Fit K-Means
    print("\n[+] Fitting K-Means with k=4...")
    clustered_df = segmenter.fit(clean_df)

    # 4. Display Cluster Characteristics
    print("\n" + "=" * 80)
    print("                      CUSTOMER SEGMENT CHARACTERISTICS")
    print("=" * 80)
    summary = segmenter.cluster_summary
    for _, row in summary.iterrows():
        print(f"\nCluster {int(row['Cluster'])}: {row['Segment_Name']}")
        print(f"  Risk Category      : {row['Risk_Level']}")
        print(f"  Customer Share     : {row['Count']} ({row['Percentage']:.1f}%)")
        print(f"  Avg Tenure         : {row['Avg_Tenure']:.1f} months")
        print(f"  Avg Monthly Charge : ${row['Avg_MonthlyCharges']:.2f}")
        print(f"  Avg Total Charge   : ${row['Avg_TotalCharges']:.2f}")
        print(f"  Avg Services Count : {row['Avg_Services']:.1f}")
        print(f"  Historical Churn   : {row['Churn_Rate']*100:.1f}%")
        print(f"  Retention Strategy : {row['Strategy']}")

    # 5. Persist artifacts
    segmenter.save("models/segmentation_kmeans.joblib")
    print("\n[+] Segmentation model saved to models/segmentation_kmeans.joblib")

    # 6. Single customer test
    sample = clean_df.iloc[0].to_dict()
    pred_res = segmenter.predict_cluster(sample)
    print("\nSample Customer Assignment:")
    print(f"  Assigned Segment: {pred_res['segment_name']} (Cluster {pred_res['cluster_id']})")
    print(f"  Tagline         : {pred_res['tagline']}")
    print(f"  Strategy        : {pred_res['recommended_strategy']}")

    print("\n>>> MODULE 3 COMPLETED SUCCESSFULLY! <<<\n")


if __name__ == "__main__":
    main()

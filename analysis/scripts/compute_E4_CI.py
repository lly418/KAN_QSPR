#!/usr/bin/env python3
"""
E4: Compute 95% confidence intervals from multi-seed results.
95% CI = mean ± 1.96 × (std / √n), n = 5 seeds
"""

import csv
import os
import math

N_SEEDS = 5
Z = 1.96  # 95% confidence
CI_FACTOR = Z / math.sqrt(N_SEEDS)  # 1.96 / 2.236 = 0.8767

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(BASE, "tables")


def compute_ci(input_path, output_path):
    rows = []
    with open(input_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mean = float(row["mean"])
            std = float(row["std"])
            half_width = CI_FACTOR * std
            ci_lower = round(mean - half_width, 4)
            ci_upper = round(mean + half_width, 4)
            rows.append({
                **row,
                "ci_95_lower": ci_lower,
                "ci_95_upper": ci_upper,
            })

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"  {output_path} ({len(rows)} rows)")


if __name__ == "__main__":
    base_results = os.path.join(BASE, "..", "results")

    # KAN multi-seed
    compute_ci(
        os.path.join(base_results, "multi_seed_summary.csv"),
        os.path.join(TABLES, "E4_KAN_confidence_intervals.csv"),
    )

    # Traditional ML baselines
    compute_ci(
        os.path.join(base_results, "baselines", "SUMMARY.csv"),
        os.path.join(TABLES, "E4_baselines_confidence_intervals.csv"),
    )

    # GNN baselines
    compute_ci(
        os.path.join(base_results, "gnn_baselines", "SUMMARY.csv"),
        os.path.join(TABLES, "E4_GNN_confidence_intervals.csv"),
    )

    print("\nDone. All 95% CI tables written to revision/tables/")

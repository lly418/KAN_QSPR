#!/usr/bin/env python3
"""
E3: Statistical significance tests — paired t-test & Wilcoxon signed-rank.
Compares KAN vs each baseline (RF, XGBoost, MLP) per dataset per metric.

Requires per-seed JSON files (generated after running revised scripts):
  - results/multi_seed_per_seed.json    (from reproduce_results.py)
  - results/baselines/per_seed.json     (from baselines/run_baselines.py)

Usage:
  python revision/scripts/compute_E3_significance.py
"""

import json
import os
import csv
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT = os.path.join(BASE, "..")

KAN_FILE = os.path.join(PROJECT, "results", "multi_seed_per_seed.json")
BASELINE_FILE = os.path.join(PROJECT, "results", "baselines", "per_seed.json")
OUTPUT = os.path.join(BASE, "tables", "E3_significance_tests.csv")

# Which baselines to compare against
BASELINE_MODELS = ["rf", "xgb", "mlp"]

# Metric direction: "higher" = larger is better, "lower" = smaller is better
METRIC_DIRECTION = {
    "roc_auc": "higher", "f1": "higher",
    "roc_auc_sym": "higher", "f1_sym": "higher",
    "rmse": "lower", "r2": "higher",
    "rmse_sym": "lower", "r2_sym": "higher",
}


def load_kan_per_seed():
    """Load KAN per-seed results. Returns {dataset: [{metric: val}, ...]}."""
    with open(KAN_FILE) as f:
        data = json.load(f)
    return data


def load_baseline_per_seed():
    """Load baseline per-seed results. Returns {(dataset, model): [{metric: val}, ...]}."""
    with open(BASELINE_FILE) as f:
        data = json.load(f)
    out = {}
    for label, seed_results in data.items():
        ds_name, model = label.split("__", 1)
        out[(ds_name, model)] = seed_results
    return out


def run_tests(kan_data, bl_data):
    rows = []
    for ds_name, kan_seeds in kan_data.items():
        # Get the list of metrics available for KAN on this dataset
        kan_metrics = [k for k in kan_seeds[0].keys()
                       if k not in ("error", "dataset", "seed", "best_params")
                       and isinstance(kan_seeds[0][k], (int, float))]

        for metric in kan_metrics:
            kan_vals = [s[metric] for s in kan_seeds if metric in s and "error" not in s]

            for model in BASELINE_MODELS:
                key = (ds_name, model)
                if key not in bl_data:
                    continue
                bl_seeds = bl_data[key]
                bl_vals = [s[metric] for s in bl_seeds
                           if metric in s and "error" not in s]

                # Need paired values — same number of seeds
                n_common = min(len(kan_vals), len(bl_vals))
                if n_common < 3:
                    continue
                kan_arr = kan_vals[:n_common]
                bl_arr = bl_vals[:n_common]

                # ── Paired t-test ──
                t_stat, t_pval = stats.ttest_rel(kan_arr, bl_arr)
                t_pval = round(float(t_pval), 6)

                # ── Wilcoxon signed-rank ──
                try:
                    w_stat, w_pval = stats.wilcoxon(kan_arr, bl_arr)
                    w_pval = round(float(w_pval), 6)
                except ValueError:
                    w_stat, w_pval = None, None  # all diffs zero or ties

                # ── Direction: is KAN better? ──
                direction = METRIC_DIRECTION.get(metric, "higher")
                kan_mean = sum(kan_arr) / len(kan_arr)
                bl_mean = sum(bl_arr) / len(bl_arr)
                if direction == "higher":
                    kan_better = kan_mean > bl_mean
                else:
                    kan_better = kan_mean < bl_mean

                # ── Significance stars ──
                sig = ""
                p_for_sig = t_pval if t_pval is not None else w_pval
                if p_for_sig is not None:
                    if p_for_sig < 0.001:
                        sig = "***"
                    elif p_for_sig < 0.01:
                        sig = "**"
                    elif p_for_sig < 0.05:
                        sig = "*"

                rows.append({
                    "dataset": ds_name,
                    "model": model,
                    "metric": metric,
                    "kan_mean": round(float(kan_mean), 4),
                    "bl_mean": round(float(bl_mean), 4),
                    "diff": round(float(kan_mean - bl_mean), 4),
                    "kan_better": kan_better,
                    "ttest_p": t_pval,
                    "wilcoxon_p": w_pval if w_pval is not None else "",
                    "significance": sig,
                    "n_pairs": n_common,
                })

    return rows


def main():
    if not os.path.exists(KAN_FILE):
        print(f"ERROR: KAN per-seed file not found: {KAN_FILE}")
        print("  Run: python reproduce_results.py --seeds 42,456,23,123,789")
        return
    if not os.path.exists(BASELINE_FILE):
        print(f"ERROR: Baseline per-seed file not found: {BASELINE_FILE}")
        print("  Run: python baselines/run_baselines.py")
        return

    print(f"Loading KAN per-seed data: {KAN_FILE}")
    kan_data = load_kan_per_seed()
    print(f"  Datasets: {list(kan_data.keys())}")

    print(f"Loading baseline per-seed data: {BASELINE_FILE}")
    bl_data = load_baseline_per_seed()
    print(f"  Entries: {len(bl_data)}")

    rows = run_tests(kan_data, bl_data)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    fieldnames = ["dataset", "model", "metric", "kan_mean", "bl_mean", "diff",
                  "kan_better", "ttest_p", "wilcoxon_p", "significance", "n_pairs"]
    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nE3 significance tests written to: {OUTPUT} ({len(rows)} rows)")

    # ── Quick summary ──
    print(f"\n{'='*80}")
    print("  SIGNIFICANCE SUMMARY: KAN vs Baselines")
    print(f"{'='*80}")
    for sig_level in ["***", "**", "*", ""]:
        subset = [r for r in rows if r["significance"] == sig_level and r["kan_better"]]
        if subset:
            label = { "***": "p<0.001", "**": "p<0.01", "*": "p<0.05", "": "n.s." }[sig_level]
            print(f"\n  KAN significantly BETTER ({label}):")
            for r in subset:
                print(f"    {r['dataset']:<12s} vs {r['model']:<6s} | {r['metric']:<12s} "
                      f"KAN={r['kan_mean']:.4f}  {r['model']}={r['bl_mean']:.4f}  p={r['ttest_p']}")

    for sig_level in ["***", "**", "*", ""]:
        subset = [r for r in rows if r["significance"] == sig_level and not r["kan_better"]]
        if subset:
            label = { "***": "p<0.001", "**": "p<0.01", "*": "p<0.05", "": "n.s." }[sig_level]
            print(f"\n  Baseline significantly BETTER ({label}):")
            for r in subset:
                print(f"    {r['dataset']:<12s} vs {r['model']:<6s} | {r['metric']:<12s} "
                      f"KAN={r['kan_mean']:.4f}  {r['model']}={r['bl_mean']:.4f}  p={r['ttest_p']}")


if __name__ == "__main__":
    main()

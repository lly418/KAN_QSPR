#!/usr/bin/env python3
"""
F1/F3: Symbolic extraction sensitivity analysis.

For 2 representative regression datasets (ESOL, FreeSolv),
runs the full pipeline with 5 seeds using fixed hyperparams from pkl,
extracts symbolic formulas from each seed, and compares formula consistency.

Usage:
  conda activate kan_fault
  python revision/scripts/compute_F1_symbolic_sensitivity.py
"""

import sys, os, copy, json, re
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from reproduce_results import (BASE_ARGS, BEST_PARAMS, _load_feats_from_cache,
                              _load_model_from_cache, load_dgl_dataset,
                              collate_molgraphs, calc_esol_descriptors,
                              calc_freesolv_descriptors)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(BASE, "output")
os.makedirs(OUTPUT, exist_ok=True)

DATASETS = ["ESOL", "FreeSolv"]
SEEDS = [42, 456, 23, 123, 789]

# Dataset-specific configs (regression)
DS_CONFIGS = {
    "ESOL": {
        "experiment_name": "esol_classification_add",
        "data_split": (80, 10, 10),
        "csv_path": "data/esol.csv",
        "alpha": 0.05, "beta": 1.5, "r2_threshold": 0.0,
        "fix_symbolic": [(0, 1, 0, "log"), (0, 2, 0, "sqrt")],
        "lib": None,  # default lib
    },
    "FreeSolv": {
        "experiment_name": "FreeSolve_add",
        "data_split": (80, 10, 10),
        "csv_path": "data/FreeSolv.csv",
        "alpha": 0.05, "beta": 1.5, "r2_threshold": 0.0,
        "fix_symbolic": [(0, 2, 0, "log")],
        "lib": ["x", "x^2", "x^3", "x^4", "exp", "log", "sqrt", "gaussian", "x^0.5", "1/x"],
    },
}


def extract_formula_text(model):
    """Extract the final symbolic formula as a human-readable string."""
    try:
        result = model.symbolic_formula()
        if isinstance(result, tuple) and len(result) >= 1:
            formulas = result[0]
            if formulas and len(formulas) > 0:
                return [str(expr) for expr in formulas if hasattr(expr, '__str__')]
        return ["(could not extract)"]
    except Exception as e:
        return [f"(error: {e})"]


def parse_formula_features(formula_texts):
    """Extract which x_i features appear in the formula."""
    features = set()
    for text in formula_texts:
        found = re.findall(r'x_(\d+)', text)
        features.update(int(f) for f in found)
    return sorted(features)


def run_regression_with_formula(dataset_name, seed):
    """Run one regression (dataset, seed) with fixed params, return formula + metrics."""
    from utils import get_data_return
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score
    from kan import KAN
    import dgl, pandas as pd
    from rdkit import Chem
    from torch.utils.data import DataLoader

    ds_cfg = DS_CONFIGS[dataset_name]
    device = torch.device("cpu")

    config = {
        **{k: v for k, v in BASE_ARGS.items()},
        "experiment_name": ds_cfg["experiment_name"],
        "frac_train": ds_cfg["data_split"][0] / 100,
        "frac_val": ds_cfg["data_split"][1] / 100,
        "frac_test": ds_cfg["data_split"][2] / 100,
        "data_split": ds_cfg["data_split"],
        "use_scaler": False,
        "ms_k": BEST_PARAMS[dataset_name]["ms_k"],
        "ms_epochs": 200, "optim": "Adam",
        "alpha": ds_cfg["alpha"], "beta": ds_cfg["beta"],
        "r2_threshold": ds_cfg["r2_threshold"], "feat_seed": 42,
        "exp_seed": seed,
    }

    # Read features + G/g_e from pkl cache
    feat_overrides = _load_feats_from_cache(config["experiment_name"], config)
    config.update(feat_overrides)
    model_overrides = _load_model_from_cache(config["experiment_name"], config)
    if model_overrides:
        config["G_override"] = model_overrides["G_override"]
        config["g_e_override"] = model_overrides["g_e_override"]
    else:
        bp = BEST_PARAMS.get(dataset_name, {})
        config["G_override"] = bp.get("G", 10)
        config["g_e_override"] = bp.get("g_e", 0.5)

    kept_feats = config["kept_feats_override"]
    G = config["G_override"]
    g_e = config["g_e_override"]
    ms_k = config.get("ms_k", 3)
    exp_seed = seed

    print(f"    Features={len(kept_feats)}, G={G}, g_e={g_e}, ms_k={ms_k}")

    # ── Load data (same pipeline as reproduce_results.py) ──
    print("    [1/4] Loading data...")
    dgl_name = "FreeSolv" if dataset_name == "FreeSolv" else dataset_name
    dataset, train_set, val_set, test_set = load_dgl_dataset(dgl_name, config)
    loader = DataLoader(dataset, batch_size=config.get("batch_size", 1),
                        collate_fn=collate_molgraphs, drop_last=True)

    all_hidden_feat, final_labels, df_masks = [], [], []
    for batch_id, batch_data in enumerate(loader):
        smiles, bg, labels, masks = batch_data
        df_masks.append(np.asarray(masks[0, :]))
        bg = dgl.add_self_loop(bg)
        mol = Chem.MolFromSmiles(smiles[0])
        # Atom features + custom descriptors (matching original scripts)
        atom_feats = bg.ndata.pop(config.get("atom_data_field", "h"))
        hidden_feat = torch.mean(atom_feats, dim=0).unsqueeze(dim=0)
        extra = calc_esol_descriptors(mol) if dataset_name == "ESOL" else calc_freesolv_descriptors(mol)
        hidden_feat = torch.cat((hidden_feat, torch.tensor(extra).unsqueeze(dim=0)), dim=1)
        all_hidden_feat.append(hidden_feat.detach().numpy().tolist())
        labels_np = labels.numpy().astype(float)
        final_labels.append(labels_np[0, 0] if labels_np.shape[1] == 1 else labels_np[0, :])

    all_hidden_feat = [item for sublist in all_hidden_feat for item in sublist]
    df = pd.DataFrame({'feat': all_hidden_feat, 'label': final_labels})
    num_columns = len(df['feat'].iloc[0])
    columns = [f'feat_{i}' for i in range(num_columns)]
    feat_df = pd.DataFrame(df['feat'].tolist(), columns=columns)
    df = df.drop(columns='feat')
    df_masks = pd.DataFrame({'masks': df_masks})
    df = pd.concat([feat_df, df, df_masks], axis=1)
    print(f"    DataFrame: {df.shape}")

    # ── Train ──
    print("    [2/4] Training...")
    scaler = StandardScaler() if config.get("use_scaler") else None
    final_data = get_data_return(df, scaler=scaler, data_split=config["data_split"],
                                 final_eval=True, feat_idxs=kept_feats,
                                 device=device, exp_seed=exp_seed)

    kan_input = final_data["train_input"].shape[1]
    model = KAN(width=[kan_input, 1], grid=G, k=ms_k,
                grid_eps=g_e, sparse_init=False, seed=exp_seed,
                auto_save=False, device=device)

    model.fit(final_data, opt="Adam", steps=200, lamb=0.0, update_grid=True,
              grid_update_num=10, stop_grid_update_step=150,
              loss_fn=torch.nn.MSELoss())

    # ── Evaluate KAN ──
    y_pred = model.forward(final_data["test_input"]).detach().cpu().numpy().flatten()
    y_test = final_data["test_label"].cpu().numpy().flatten()
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # ── Symbolify + extract formula ──
    print("    [3/4] Symbolification...")
    for (layer, in_idx, out_idx, func) in ds_cfg["fix_symbolic"]:
        model.fix_symbolic(layer, in_idx, out_idx, func)
    kwargs = {"verbose": 0, "alpha": ds_cfg["alpha"], "beta": ds_cfg["beta"],
              "r2_threshold": ds_cfg["r2_threshold"]}
    if ds_cfg.get("lib"):
        kwargs["lib"] = ds_cfg["lib"]
    model.auto_symbolic(**kwargs)
    formula_texts = extract_formula_text(model)

    # ── Evaluate symbolic ──
    print("    [4/4] Evaluating symbolic...")
    y_pred_sym = model.forward(final_data["test_input"]).detach().cpu().numpy().flatten()
    rmse_sym = np.sqrt(mean_squared_error(y_test, y_pred_sym))
    r2_sym = r2_score(y_test, y_pred_sym)

    return {
        "formula": formula_texts,
        "rmse": round(float(rmse), 4), "r2": round(float(r2), 4),
        "rmse_sym": round(float(rmse_sym), 4), "r2_sym": round(float(r2_sym), 4),
        "features_used": parse_formula_features(formula_texts),
        "G": G, "g_e": g_e,
    }


def analyze_consistency(dataset_name, seed_results):
    """Analyze formula consistency across seeds."""
    valid = [r for r in seed_results if "rmse_sym" in r]
    if not valid:
        print(f"\n  ALL SEEDS FAILED — cannot analyze consistency.")
        return
    n_seeds = len(seed_results)
    n_valid = len(valid)

    print(f"\n{'='*80}")
    print(f"  {dataset_name}: FORMULA CONSISTENCY (valid: {n_valid}/{n_seeds} seeds)")
    print(f"{'='*80}")

    # 1. Performance stability
    rmse_vals = [r["rmse_sym"] for r in valid]
    r2_vals = [r["r2_sym"] for r in valid]
    print(f"\n  Symbolic performance:")
    print(f"    RMSE: {np.mean(rmse_vals):.4f} ± {np.std(rmse_vals):.4f}")
    print(f"    R²:   {np.mean(r2_vals):.4f} ± {np.std(r2_vals):.4f}")
    print(f"    RMSE CV: {np.std(rmse_vals)/np.mean(rmse_vals)*100:.1f}%")

    # 2. KAN vs Symbolic performance gap
    rmse_kan_vals = [r["rmse"] for r in valid]
    print(f"\n  KAN → Symbolic degradation:")
    print(f"    RMSE Δ: {np.mean(rmse_vals)-np.mean(rmse_kan_vals):+.4f}")
    r2_kan_vals = [r["r2"] for r in valid]
    print(f"    R² Δ:   {np.mean(r2_vals)-np.mean(r2_kan_vals):+.4f}")

    # 3. Feature frequency across seeds
    all_features = []
    for r in valid:
        all_features.extend(r["features_used"])
    feature_counts = Counter(all_features)
    print(f"\n  Feature occurrence across {n_valid} valid seeds:")
    for feat_idx in sorted(feature_counts.keys()):
        count = feature_counts[feat_idx]
        bar = "█" * count + "░" * (n_valid - count)
        print(f"    feat_{feat_idx}: {bar} ({count}/{n_valid})")

    # 4. Formula text for each seed
    print(f"\n  Extracted formulas:")
    for i, r in enumerate(seed_results):
        if "formula" not in r:
            print(f"    seed={SEEDS[i]}: ERROR - {r.get('error', 'unknown')}")
            continue
        for j, formula in enumerate(r["formula"]):
            if len(formula) > 300:
                formula = formula[:300] + "..."
            print(f"    seed={SEEDS[i]}: {formula}")

    # 5. Core vs transient features
    core = [f for f, c in feature_counts.items() if c >= 3]
    transient = [f for f, c in feature_counts.items() if c < 3]
    print(f"\n  Core features (≥3/{n_valid} seeds):   {core if core else 'none'}")
    print(f"  Transient features (<3/{n_valid}):     {transient if transient else 'none'}")
    print(f"  Consistency score:             {len(core)}/{len(core)+len(transient)} features stable ({len(core)/max(len(core)+len(transient),1)*100:.0f}%)")


def main():
    print(f"{'='*60}")
    print(f"  F1/F3: SYMBOLIC EXTRACTION SENSITIVITY")
    print(f"  Datasets: {DATASETS}, Seeds: {SEEDS}")
    print(f"{'='*60}")

    all_data = {}

    for ds in DATASETS:
        print(f"\n── {ds} ──")
        seed_results = []
        for seed in SEEDS:
            print(f"  seed={seed}:", end="", flush=True)
            try:
                result = run_regression_with_formula(ds, seed)
                seed_results.append(result)
                print(f" rmse_sym={result['rmse_sym']:.4f}, features={result['features_used']}")
            except Exception as e:
                print(f" ERROR: {e}")
                import traceback
                traceback.print_exc()
                seed_results.append({"error": str(e), "seed": seed})

        all_data[ds] = seed_results
        analyze_consistency(ds, seed_results)

    # Save
    output_path = os.path.join(OUTPUT, "F1_symbolic_sensitivity.json")
    serializable = {}
    for ds, results in all_data.items():
        serializable[ds] = []
        for r in results:
            clean = {}
            for k, v in r.items():
                if isinstance(v, (int, float, str, bool, list, type(None))):
                    clean[k] = v
                elif isinstance(v, np.integer):
                    clean[k] = int(v)
                elif isinstance(v, np.floating):
                    clean[k] = float(v)
                else:
                    clean[k] = str(v)
            serializable[ds].append(clean)

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\n\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

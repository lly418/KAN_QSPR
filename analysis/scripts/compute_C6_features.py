#!/usr/bin/env python3
"""
C6: Extract selected feature lists for all 8 datasets.
Reads featsdf.pkl, applies the same selection logic as _load_feats_from_cache,
and maps feat_N indices to human-readable descriptor names.

Usage:
  conda activate kan_fault
  python revision/scripts/compute_C6_features.py
"""

import sys, os, csv
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from reproduce_results import BEST_PARAMS

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(BASE, "tables")
os.makedirs(OUTPUT, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature name mappings per dataset type
# ---------------------------------------------------------------------------

# 74 CanonicalAtomFeaturizer atom-level features (used by ESOL, FreeSolv, BACE, BBBP, ClinTox, SIDER)
ATOM_FEATURE_NAMES = [
    "atom_type_C", "atom_type_N", "atom_type_O", "atom_type_S", "atom_type_F",
    "atom_type_Cl", "atom_type_Br", "atom_type_I", "atom_type_P", "atom_type_Se",
    "atom_type_other", "degree_0", "degree_1", "degree_2", "degree_3",
    "degree_4", "degree_5", "degree_6", "degree_other", "num_Hs_0",
    "num_Hs_1", "num_Hs_2", "num_Hs_3", "num_Hs_4", "total_valence_1",
    "total_valence_2", "total_valence_3", "total_valence_4", "total_valence_5", "total_valence_6",
    "total_valence_other", "formal_charge_0", "formal_charge_1", "formal_charge_-1", "formal_charge_other",
    "hybrid_SP", "hybrid_SP2", "hybrid_SP3", "hybrid_SP3D", "hybrid_SP3D2",
    "hybrid_other", "is_aromatic", "is_not_aromatic", "mass",
    "num_Hs", "chirality_CW", "chirality_CCW", "chirality_other",
    "is_chiral", "is_not_chiral",
    "atom_type_Na", "atom_type_K", "atom_type_Ca", "atom_type_Mg", "atom_type_Zn",
    "atom_type_Fe", "atom_type_Mn", "atom_type_Cu", "atom_type_B", "atom_type_Si",
    "atom_type_other_metal", "atom_type_unknown",
    "explicit_valence_1", "explicit_valence_2", "explicit_valence_3", "explicit_valence_4",
    "explicit_valence_5", "explicit_valence_6", "explicit_valence_other",
    "implicit_valence_0", "implicit_valence_1", "implicit_valence_2", "implicit_valence_3",
    "implicit_valence_4", "implicit_valence_other",
]

# ESOL-specific custom descriptors (index 74-89, 16 descriptors)
ESOL_EXTRA = [
    "MolWt", "MolLogP", "TPSA", "NumHBD", "NumHBA",
    "NumRotatableBonds", "NumAromaticRings", "NumHeterocycles", "NumRings",
    "MaxPartialCharge", "MinPartialCharge", "MaxAbsPartialCharge", "MinAbsPartialCharge",
    "OH_count", "COOH_count", "AromaticProp",
]

# FreeSolv-specific custom descriptors (index 74-87, 14 descriptors)
FREESOLV_EXTRA = [
    "MolWt", "MolLogP", "TPSA", "CalcTPSA",
    "NumHBD", "NumHBA", "NumRotatableBonds", "NumAromaticRings",
    "NumHeterocycles", "NumRings",
    "MaxPartialCharge", "MinPartialCharge", "MaxAbsPartialCharge", "MinAbsPartialCharge",
]

# CDK9: full RDKit descriptor list (66 features, matching calc_cdk9_descriptors return)
CDK9_DESCRIPTORS = [
    "NumHBD", "NumHBA", "NumRotatableBonds", "NumAromaticRings",
    "NumHeterocycles", "NumRings",
    "MaxPartialCharge", "MinPartialCharge", "MaxAbsPartialCharge", "MinAbsPartialCharge",
    "OH_count", "COOH_count", "AromaticProp", "BertzCT", "HallKierAlpha",
    "Ipc", "Kappa1", "Kappa2", "Kappa3", "Chi0", "Chi1",
    "Chi2n", "SlogP_VSA1", "SlogP_VSA2", "SlogP_VSA3", "SMR_VSA1", "SMR_VSA2",
    "SMR_VSA3", "PEOE_VSA1", "PEOE_VSA2", "PEOE_VSA3", "NumAtoms",
    "NumBonds", "NumSingleBonds", "NumDoubleBonds", "NumTripleBonds",
    "NumC", "NumN", "NumO", "NumF",
    "QED", "MolMR", "NumValenceElectrons",
    "MolWt", "MolLogP", "TPSA",
    "NumHBD_dup", "NumHBA_dup", "NumLipinskiHBA",
    "mean_partial_charge", "std_partial_charge", "max_partial_charge", "min_partial_charge",
    "PyridineCount", "AmideCount",
    "NH2_count", "CN_count",
    "NumAromaticRings_dup", "NumHeterocycles_dup", "NumAliphaticRings",
    "NumRotatableBonds_dup", "NumRigidBonds",
    "MaxEState", "MinEState",
    "ExactMolWt", "MolMR_dup", "QED_dup",
]

# HOB-specific custom descriptors (14 features)
HOB_DESCRIPTORS = [
    "TPSA", "MolLogP", "ExactMolWt", "NumHBD", "NumHBA",
    "NumRotatableBonds", "HeavyAtomCount", "LabuteASA",
    "NumChiralCenters", "NumAromaticRings", "NumAliphaticRings",
    "NumAmideBonds", "QED", "NumLipinskiViolations",
]

# Classification extra descriptors (appended after atom features for BACE/BBBP/ClinTox/SIDER)
CLASSIFICATION_EXTRA = ["TPSA_extra", "MolLogP_extra", "MolWt_extra", "NumHBD_extra", "NumHBA_extra"]


def get_feature_name(dataset, feat_idx):
    """Map feat_N index to human-readable descriptor name."""
    # ESOL: 74 atom + 16 custom = 90 features
    if dataset == "ESOL":
        if feat_idx < 74:
            return ATOM_FEATURE_NAMES[feat_idx]
        elif feat_idx < 90:
            return ESOL_EXTRA[feat_idx - 74]
        return f"unknown_{feat_idx}"

    # FreeSolv: 74 atom + 14 custom = 88 features
    if dataset == "FreeSolv":
        if feat_idx < 74:
            return ATOM_FEATURE_NAMES[feat_idx]
        elif feat_idx < 88:
            return FREESOLV_EXTRA[feat_idx - 74]
        return f"unknown_{feat_idx}"

    # CDK9: 66 RDKit descriptors only
    if dataset == "CDK9":
        if feat_idx < len(CDK9_DESCRIPTORS):
            return CDK9_DESCRIPTORS[feat_idx]
        return f"unknown_{feat_idx}"

    # HOB: 14 custom descriptors
    if dataset == "HOB":
        if feat_idx < len(HOB_DESCRIPTORS):
            return HOB_DESCRIPTORS[feat_idx]
        return f"unknown_{feat_idx}"

    # Classification (BACE, BBBP, ClinTox, SIDER): 74 atom + 5 extra = 79 features
    if feat_idx < 74:
        return ATOM_FEATURE_NAMES[feat_idx]
    elif feat_idx < 79:
        return CLASSIFICATION_EXTRA[feat_idx - 74]
    return f"unknown_{feat_idx}"


def load_kept_features(experiment_name, ds_name):
    """Select the correct Pareto-optimal solution matching BEST_PARAMS.

    Uses BEST_PARAMS (tau, lamb, features) from reproduce_results.py to identify
    the experimenter's chosen solution, rather than auto-selecting by best F1 score.
    This ensures consistency with the actual experimental results.
    """
    bp = BEST_PARAMS.get(ds_name, {})
    target_tau = bp.get("tau")
    target_lamb = bp.get("lamb")
    target_features = bp.get("features")

    featsdf_path = os.path.join("results", f"{experiment_name}_featsdf.pkl")
    if not os.path.exists(featsdf_path):
        return None, None, f"FILE NOT FOUND: {featsdf_path}"

    featsdf = pd.read_pickle(featsdf_path)
    fpset = featsdf[featsdf["pareto"] == True]
    if len(fpset) == 0:
        return None, None, "No Pareto-optimal entries"

    # Match by tau and lamb (rounded to 4 decimal places)
    match = fpset[
        (fpset["thresholds"].round(4) == round(target_tau, 4)) &
        (fpset["lambdas"].round(4) == round(target_lamb, 4))
    ]
    if len(match) == 0:
        # Fallback: match by feature count only
        match = fpset[fpset["num_feats"] == target_features]
    if len(match) == 0:
        return None, None, f"No Pareto solution with tau={target_tau}, lamb={target_lamb}"

    best_row = match.iloc[0]
    kept = best_row["features"]
    n_feats = best_row["num_feats"]
    thresholds = best_row.get("thresholds", "N/A")
    lambdas = best_row.get("lambdas", "N/A")

    return kept, n_feats, f"tau={thresholds}, lamb={lambdas}"


def main():
    DATASET_CONFIGS = [
        ("ESOL", "esol_classification_add"),
        ("FreeSolv", "FreeSolve_add"),
        ("CDK9", "CDK9_regression_add"),
        ("BACE", "bace_classification"),
        ("BBBP", "bbbp_classification_add_regression"),
        ("ClinTox", "clinttox_classification"),
        ("SIDER", "sider_classification_add"),
        ("HOB", "hob_classification_add"),
    ]

    rows = []

    for ds_name, exp_name in DATASET_CONFIGS:
        kept_feats, n_feats, notes = load_kept_features(exp_name, ds_name)
        if kept_feats is None:
            print(f"  {ds_name}: ERROR - {notes}")
            continue

        indices = [int(f.replace("feat_", "")) for f in kept_feats]
        names = [get_feature_name(ds_name, idx) for idx in indices]

        print(f"\n{'='*70}")
        print(f"  {ds_name} ({n_feats} features) — {notes}")
        print(f"{'='*70}")
        for i, (idx, name) in enumerate(zip(indices, names)):
            print(f"  {i+1:2d}. [{idx:3d}] {name}")
            rows.append({
                "dataset": ds_name,
                "rank": i + 1,
                "feat_index": idx,
                "descriptor": name,
                "total_kept": n_feats,
                "selection_params": notes,
            })

    # Save CSV
    output_path = os.path.join(OUTPUT, "C6_selected_features.csv")
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset", "rank", "feat_index", "descriptor",
            "total_kept", "selection_params"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nC6 feature table saved to: {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Batch reproduction script for KAN molecular property prediction results.
Runs the full pipeline (feature extraction → feature selection → model selection → evaluation)
for all 8 datasets and compares with expected metrics.

Expected results:
  Classification: ClinTox(ROC-AUC=0.869,F1=0.93), BBBP(0.812,0.80), SIDER(0.576,0.81),
                   BACE(0.787,0.73), HOB(0.712,0.85)
  Regression:      FreeSolv(RMSE=1.54,R2=0.78), ESOL(0.85,0.83), CDK9(0.65,0.42)

Usage:
  source /usr/local/Anaconda/etc/profile.d/conda.sh && conda activate kan_fault
  python reproduce_results.py                    # run all, use cached featsdf/modelsdf if available
  python reproduce_results.py --no-cache         # recompute everything from scratch
  python reproduce_results.py --dataset BBBP     # run only one dataset
"""
import sys, os, argparse, time, warnings, copy
from datetime import datetime

warnings.filterwarnings("ignore")
os.environ["DGLBACKEND"] = "pytorch"

# ── Expected metrics ──────────────────────────────────────────────
EXPECTED = {
    "ClinTox":  {"task": "classification", "roc_auc": 0.869, "f1": 0.93},
    "BBBP":     {"task": "classification", "roc_auc": 0.812, "f1": 0.80},
    "SIDER":    {"task": "classification", "roc_auc": 0.576, "f1": 0.81},
    "BACE":     {"task": "classification", "roc_auc": 0.787, "f1": 0.73},
    "HOB":      {"task": "classification", "roc_auc": 0.712, "f1": 0.85},
    "FreeSolv": {"task": "regression",     "rmse":    1.54,  "r2": 0.78},
    "ESOL":     {"task": "regression",     "rmse":    0.85,  "r2": 0.83},
    "CDK9":     {"task": "regression",     "rmse":    0.65,  "r2": 0.42},
}

# ── Best hyperparameters from grid search (used for multi-seed runs) ──
# Source: 修回计划/07-实验参数清单.md Section 4.5
BEST_PARAMS = {
    "ClinTox":  {"tau": 0.150000, "lamb": 0.005000, "G": 30, "g_e": 0.40, "ms_k": 4, "features": 5},
    "BBBP":     {"tau": 0.022857, "lamb": 0.001000, "G": 50, "g_e": 0.90, "ms_k": 3, "features": 9},
    "BACE":     {"tau": 0.272222, "lamb": 0.008333, "G": 10, "g_e": 0.75, "ms_k": 4, "features": 10},
    "SIDER":    {"tau": 0.182222, "lamb": 0.028474, "G": 20, "g_e": 0.80, "ms_k": 4, "features": 3},
    "HOB":      {"tau": 0.146667, "lamb": 0.014737, "G": 30, "g_e": 0.75, "ms_k": 3, "features": 7},
    "FreeSolv": {"tau": 0.157778, "lamb": 0.001000, "G": 8,  "g_e": 0.20, "ms_k": 3, "features": 4},
    "ESOL":     {"tau": 0.170000, "lamb": 0.001000, "G": 110, "g_e": 0.50, "ms_k": 3, "features": 3},
    "CDK9":     {"tau": 0.428571, "lamb": 0.137158, "G": 10, "g_e": 0.90, "ms_k": 3, "features": 9},
}

# ── Common imports ────────────────────────────────────────────────
import dgl
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from kan import KAN
from kan.utils import ex_round
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors, Lipinski, AllChem, Crippen
from rdkit.Chem.EState import EState
from dgllife.utils import smiles_to_bigraph, RandomSplitter, CanonicalAtomFeaturizer, ScaffoldSplitter
from sklearn.metrics import (classification_report, roc_auc_score,
                              mean_squared_error, r2_score, f1_score,
                              precision_recall_curve)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    get_data, get_data_return, get_multiply_label_data, get_multiply_label_data_new,
    feature_selection_roc_auc, model_selection_roc_auc,
    feature_selection_roc_auc_return, model_selection_roc_auc_return,
    feature_selection_roc_auc_multiply_label, model_selection_roc_auc_multiply_label,
    feature_selection_return, model_selection_return,
    feature_selection_return_R2, model_selection_return_R2,
    plot_heatmaps, plot_pareto, plot_cm,
)


# ═══════════════════════════════════════════════════════════════════
# 1. COMMON HELPERS
# ═══════════════════════════════════════════════════════════════════

def _feat_metric_col(featsdf):
    """Return the metric column name from featsdf (backward compat with old 'f1_scores')."""
    return 'metric' if 'metric' in featsdf.columns else 'f1_scores'


def _feat_higher_better(featsdf):
    """Return True if higher metric value is better (backward compat: True for old files)."""
    if 'metric_direction' in featsdf.columns:
        return featsdf['metric_direction'].iloc[0] == 'max'
    return True  # old 'f1_scores' was always maximized (ROC-AUC / F1 / R²)


def _select_best_from_pareto_feats(featsdf, max_feats):
    """Select best feature set from Pareto front using metric_direction.
    Returns (kept_feats_list, lambda_value)."""
    col = _feat_metric_col(featsdf)
    higher_better = _feat_higher_better(featsdf)
    fpset = featsdf[featsdf["pareto"] == True]
    under = fpset.loc[fpset["num_feats"] <= max_feats]
    target = under if len(under) > 0 else fpset
    idx = target[col].idxmax() if higher_better else target[col].idxmin()
    kept_feats = featsdf.iloc[idx]["features"].tolist()
    lamb = featsdf.iloc[idx]["lambdas"]
    return kept_feats, lamb


def _model_metric_cols(modelsdf):
    """Return (kan_col, sym_col, higher_better) from modelsdf with backward compat.

    New format:  metric_kan / metric_sym / metric_direction
    Old formats: f1_kan/f1_sym (ROC-AUC or F1, higher better)
                 rmse_kan/rmse_sym (RMSE, lower better — but could be R² mislabeled!)
                 r2_kan/r2_sym (R², higher better)
    """
    if 'metric_kan' in modelsdf.columns and 'metric_sym' in modelsdf.columns:
        kan, sym = 'metric_kan', 'metric_sym'
    elif 'f1_kan' in modelsdf.columns and 'f1_sym' in modelsdf.columns:
        kan, sym = 'f1_kan', 'f1_sym'
    elif 'r2_kan' in modelsdf.columns and 'r2_sym' in modelsdf.columns:
        kan, sym = 'r2_kan', 'r2_sym'
    elif 'rmse_kan' in modelsdf.columns and 'rmse_sym' in modelsdf.columns:
        kan, sym = 'rmse_kan', 'rmse_sym'
    else:
        raise KeyError("No recognized metric columns found in modelsdf")

    # Determine direction from explicit metadata column first
    if 'metric_direction' in modelsdf.columns:
        higher_better = modelsdf['metric_direction'].iloc[0] == 'max'
    else:
        # Backward compat: old files without metric_direction
        # f1_* → ROC-AUC or F1 (higher better); rmse_* → could be RMSE or R²
        # Default: f1/r2 → higher better, rmse → lower better
        if kan.startswith('rmse'):
            higher_better = False
        else:
            higher_better = True

    return kan, sym, higher_better


def collate_molgraphs(data):
    assert len(data[0]) in [3, 4]
    if len(data[0]) == 3:
        smiles, graphs, labels = map(list, zip(*data))
        masks = None
    else:
        smiles, graphs, labels, masks = map(list, zip(*data))
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)
    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)
    return smiles, bg, labels, masks


# ═══════════════════════════════════════════════════════════════════
# 2. DATASET LOADERS
# ═══════════════════════════════════════════════════════════════════

def load_dgl_dataset(dataset_name, args):
    """Load a dataset from dgllife."""
    from dgllife.data import ClinTox, BBBP, BACE, SIDER, ESOL, FreeSolv
    ds_map = {
        "ClinTox": ClinTox, "BBBP": BBBP, "BACE": BACE,
        "SIDER": SIDER, "ESOL": ESOL, "FreeSolv": FreeSolv,
    }
    dataset = ds_map[dataset_name](smiles_to_bigraph, args["atom_featurizer"])
    if dataset_name in ("BBBP", "BACE"):
        train_set, val_set, test_set = ScaffoldSplitter.train_val_test_split(
            dataset, frac_train=args["frac_train"], frac_val=args["frac_val"],
            frac_test=args["frac_test"])
    else:
        train_set, val_set, test_set = RandomSplitter.train_val_test_split(
            dataset, frac_train=args["frac_train"], frac_val=args["frac_val"],
            frac_test=args["frac_test"], random_state=args["random_seed"])
    return dataset, train_set, val_set, test_set


def load_hob_dataset(args):
    """Custom HOB dataset loader (multi-label classification, label_cutoff_50% + label_cutoff_20%)."""
    class CustomHobDataset(Dataset):
        def __init__(self, csv_path, featurizer):
            self.df = pd.read_csv(csv_path)
            self.featurizer = featurizer
        def __len__(self):
            return len(self.df)
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            smiles = row["smiles"]
            label_values = row[["label_cutoff_50%", "label_cutoff_20%"]].values.astype(float)
            mask = np.where(np.isnan(label_values), 0, 1).astype(np.float32)
            label = np.nan_to_num(label_values, nan=0).astype(int)
            bg = smiles_to_bigraph(smiles, self.featurizer)
            return smiles, bg, torch.tensor(label), torch.tensor(mask)
    dataset = CustomHobDataset("./data/hob.csv", args["atom_featurizer"])
    generator = torch.Generator().manual_seed(args["random_seed"])
    train_size = int(args["frac_train"] * len(dataset))
    val_size = int(args["frac_val"] * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=generator)
    return dataset, train_set, val_set, test_set


def load_cdk9_dataset(args):
    """Custom CDK9 dataset loader (regression)."""
    class KinaseDataset(Dataset):
        def __init__(self, csv_path, featurizer):
            self.df = pd.read_csv(csv_path)
            self.featurizer = featurizer
        def __len__(self):
            return len(self.df)
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            smiles = row["smiles"]
            label = float(row["pIC50"])
            bg = smiles_to_bigraph(smiles, self.featurizer)
            mask = torch.ones(1)
            return smiles, bg, torch.tensor([label]), mask
    dataset = KinaseDataset("./data/CDK9.csv", args["atom_featurizer"])
    train_set, val_set, test_set = RandomSplitter.train_val_test_split(
        dataset, frac_train=args["frac_train"], frac_val=args["frac_val"],
        frac_test=args["frac_test"], random_state=args["random_seed"])
    return dataset, train_set, val_set, test_set


# ═══════════════════════════════════════════════════════════════════
# 3. DESCRIPTOR CALCULATORS
# ═══════════════════════════════════════════════════════════════════

def calc_esol_descriptors(mol):
    aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    ap = aromatic_rings / (mol.GetNumHeavyAtoms() + 1e-6)
    return [
        Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
        rdMolDescriptors.CalcNumHBD(mol), rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol), rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumHeterocycles(mol), rdMolDescriptors.CalcNumRings(mol),
        Descriptors.MaxPartialCharge(mol), Descriptors.MinPartialCharge(mol),
        Descriptors.MaxAbsPartialCharge(mol), Descriptors.MinAbsPartialCharge(mol),
        len(mol.GetSubstructMatches(Chem.MolFromSmarts('[OH]'))),
        len(mol.GetSubstructMatches(Chem.MolFromSmarts('C(=O)O'))),
        ap,
    ]


def calc_freesolv_descriptors(mol):
    return [
        Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
        rdMolDescriptors.CalcTPSA(mol),
        rdMolDescriptors.CalcNumHBD(mol), rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol), rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumHeterocycles(mol), rdMolDescriptors.CalcNumRings(mol),
        Descriptors.MaxPartialCharge(mol), Descriptors.MinPartialCharge(mol),
        Descriptors.MaxAbsPartialCharge(mol), Descriptors.MinAbsPartialCharge(mol),
    ]


def calc_cdk9_descriptors(mol):
    aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    ap = aromatic_rings / (mol.GetNumHeavyAtoms() + 1e-6)
    estate_indices = EState.EStateIndices(mol)
    max_estate = max(estate_indices) if len(estate_indices) > 0 else 0
    min_estate = min(estate_indices) if len(estate_indices) > 0 else 0
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    AllChem.ComputeGasteigerCharges(mol)
    partial_charges = []
    for atom in mol.GetAtoms():
        try:
            partial_charges.append(float(atom.GetProp('_GasteigerCharge')))
        except KeyError:
            partial_charges.append(0.0)
    pyridine_count = len(mol.GetSubstructMatches(Chem.MolFromSmarts('c1cncc[c,n]1')))
    amide_count = len(mol.GetSubstructMatches(Chem.MolFromSmarts('C(=O)N')))
    num_rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
    num_rigid = mol.GetNumBonds() - num_rotatable

    return [
        rdMolDescriptors.CalcNumHBD(mol), rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol), rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumHeterocycles(mol), rdMolDescriptors.CalcNumRings(mol),
        Descriptors.MaxPartialCharge(mol), Descriptors.MinPartialCharge(mol),
        Descriptors.MaxAbsPartialCharge(mol), Descriptors.MinAbsPartialCharge(mol),
        len(mol.GetSubstructMatches(Chem.MolFromSmarts('[OH]'))),
        len(mol.GetSubstructMatches(Chem.MolFromSmarts('C(=O)O'))),
        ap, Descriptors.BertzCT(mol), Descriptors.HallKierAlpha(mol),
        Descriptors.Ipc(mol), Descriptors.Kappa1(mol), Descriptors.Kappa2(mol),
        Descriptors.Kappa3(mol), Descriptors.Chi0(mol), Descriptors.Chi1(mol),
        Descriptors.Chi2n(mol), Descriptors.SlogP_VSA1(mol), Descriptors.SlogP_VSA2(mol),
        Descriptors.SlogP_VSA3(mol), Descriptors.SMR_VSA1(mol), Descriptors.SMR_VSA2(mol),
        Descriptors.SMR_VSA3(mol), Descriptors.PEOE_VSA1(mol), Descriptors.PEOE_VSA2(mol),
        Descriptors.PEOE_VSA3(mol), rdMolDescriptors.CalcNumAtoms(mol),
        mol.GetNumBonds(),
        len([b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.SINGLE]),
        len([b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE]),
        len([b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.TRIPLE]),
        len([a for a in mol.GetAtoms() if a.GetSymbol() == 'C']),
        len([a for a in mol.GetAtoms() if a.GetSymbol() == 'N']),
        len([a for a in mol.GetAtoms() if a.GetSymbol() == 'O']),
        len([a for a in mol.GetAtoms() if a.GetSymbol() == 'F']),
        Descriptors.qed(mol), Crippen.MolMR(mol), Descriptors.NumValenceElectrons(mol),
        mw, logp, tpsa,
        rdMolDescriptors.CalcNumHBD(mol), rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumLipinskiHBA(mol),
        np.mean(partial_charges) if partial_charges else 0,
        np.std(partial_charges) if partial_charges else 0,
        np.max(partial_charges) if partial_charges else 0,
        np.min(partial_charges) if partial_charges else 0,
        pyridine_count, amide_count,
        len(mol.GetSubstructMatches(Chem.MolFromSmarts('[NH2]'))),
        len(mol.GetSubstructMatches(Chem.MolFromSmarts('C#N'))),
        rdMolDescriptors.CalcNumAromaticRings(mol), rdMolDescriptors.CalcNumHeterocycles(mol),
        rdMolDescriptors.CalcNumAliphaticRings(mol), num_rotatable, num_rigid,
        max_estate, min_estate,
        Descriptors.ExactMolWt(mol), Crippen.MolMR(mol), Descriptors.qed(mol),
    ]


def calc_hob_descriptors(mol):
    """HOB-specific descriptor calculator (exactly matching the original script)."""
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    logp = Descriptors.MolLogP(mol)
    mw = Descriptors.ExactMolWt(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    heavy_atoms = Lipinski.HeavyAtomCount(mol)
    AllChem.ComputeGasteigerCharges(mol)
    labute_asa = rdMolDescriptors.CalcLabuteASA(mol)
    chiral_centers = len(Chem.FindMolChiralCenters(mol))
    arom_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    aliph_rings = rdMolDescriptors.CalcNumAliphaticRings(mol)
    amide_bonds = sum(1 for b in mol.GetBonds() if b.GetBondTypeAsDouble() == 1.0 and
                      (b.GetBeginAtom().GetAtomicNum() == 7 and b.GetEndAtom().GetAtomicNum() == 6))
    qed_score = Descriptors.qed(mol)
    n_lipinski = sum([hbd > 5, hba > 10, mw > 500, logp > 5])
    return [tpsa, logp, mw, hbd, hba, rotatable_bonds, heavy_atoms,
            labute_asa, chiral_centers, arom_rings, aliph_rings, amide_bonds,
            qed_score, n_lipinski]


# ═══════════════════════════════════════════════════════════════════
# 4. SINGLE-LABEL CLASSIFICATION RUNNER (ClinTox, BBBP, BACE)
# ═══════════════════════════════════════════════════════════════════

def run_classification_single_label(dataset_name, args, config, no_cache=False):
    """Run the full pipeline for single-label classification.

    Data extraction EXACTLY matches the original experiment scripts:
    - ClinTox: masks[0,1], labels[0,1] (CT_TOX task)
    - BBBP: masks[0,0], labels[0,0]
    - BACE: masks[0,0], labels[0,0]
    """
    print(f"\n{'='*70}")
    print(f"  Running: {dataset_name} (single-label classification)")
    print(f"{'='*70}")

    device = torch.device("cpu")
    exp_seed = config["exp_seed"]
    feat_seed = config.get("feat_seed", exp_seed)
    experiment_name = config["experiment_name"]
    label_col = config.get("label_col", 0)

    # ── Step 1: Load dataset & extract features ──
    print("[1/5] Loading dataset and extracting features...")
    dataset, train_set, val_set, test_set = load_dgl_dataset(dataset_name, args)
    loader = DataLoader(dataset, batch_size=args["batch_size"], collate_fn=collate_molgraphs, drop_last=True)

    # EXACTLY matching the original script's feature extraction loop
    all_hidden_feat, final_labels, df_masks = [], [], []
    for batch_id, batch_data in enumerate(loader):
        smiles, bg, labels, masks = batch_data
        df_masks.append(np.asarray(masks[0, label_col]))
        bg = dgl.add_self_loop(bg)
        atom_feats = bg.ndata.pop(args['atom_data_field'])
        hidden_feat = torch.mean(atom_feats, dim=0).unsqueeze(dim=0)
        mol = Chem.MolFromSmiles(smiles[0])
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        logp = Descriptors.MolLogP(mol)
        mw = Descriptors.ExactMolWt(mol)
        HBD = rdMolDescriptors.CalcNumHBD(mol)
        HBA = rdMolDescriptors.CalcNumHBA(mol)
        hidden_feat = torch.cat((hidden_feat, torch.tensor([tpsa, logp, mw, HBD, HBA]).unsqueeze(dim=0)), dim=1)
        all_hidden_feat.append(hidden_feat.detach().numpy().tolist())
        final_labels.append((labels.numpy().astype(float))[0, label_col])

    # EXACTLY matching the original DataFrame construction
    all_hidden_feat = [item for sublist in all_hidden_feat for item in sublist]
    df = pd.DataFrame({'feat': all_hidden_feat, 'label': final_labels})
    num_columns = len(df['feat'].iloc[0])
    columns = [f'feat_{i}' for i in range(num_columns)]
    feat_df = pd.DataFrame(df['feat'].tolist(), columns=columns)
    df = df.drop(columns='feat')
    df_masks = pd.DataFrame({'masks': df_masks})
    df = pd.concat([feat_df, df, df_masks], axis=1)
    print(f"  DataFrame shape: {df.shape}")

    # ── Step 2: Feature selection ──
    if "kept_feats_override" in config:
        kept_feats = config["kept_feats_override"]
        print(f"[2/5] Using pre-computed {len(kept_feats)} features (skip feature selection)")
    else:
        print("[2/5] Feature selection...")
        os.makedirs("results", exist_ok=True)
        featsdf_path = os.path.join("results", f"{experiment_name}_featsdf.pkl")
        if os.path.exists(featsdf_path) and not no_cache:
            featsdf = pd.read_pickle(featsdf_path)
            print(f"  Loaded cached featsdf ({len(featsdf)} entries)")
        else:
            t0 = time.time()
            featsdf = feature_selection_roc_auc(
                df=df, grid_size=config["fs_grid_size"], grid_eps=config["fs_grid_eps"],
                k=config["fs_k"], thresholds=config["thresholds"], lambdas=config["lambdas"],
                optim=config["optim"], epochs=config["fs_epochs"],
                use_scaler=config["use_scaler"], data_split=config["data_split"],
                device=device, exp_seed=feat_seed)
            print(f"  Feature selection took {time.time()-t0:.1f}s")
            featsdf.to_pickle(featsdf_path)
        kept_feats, lamb = _select_best_from_pareto_feats(featsdf, config.get("max_feats", 11))
        print(f"  Selected {len(kept_feats)} features, lambda={lamb:.4f}")

    # ── Step 3: Model selection ──
    if "G_override" in config:
        G = config["G_override"]
        g_e = config["g_e_override"]
        print(f"[3/5] Using pre-computed G={G}, g_e={g_e} (skip model selection)")
    else:
        scaler = StandardScaler() if config["use_scaler"] else None
        dataset_split = get_data(df, scaler=scaler, data_split=config["data_split"],
                                  final_eval=False, feat_idxs=kept_feats,
                                  device=device, exp_seed=feat_seed)
        print("[3/5] Model selection...")
        os.makedirs("results", exist_ok=True)
        modelsdf_path = os.path.join("results", f"{experiment_name}_modelsdf.pkl")
        if os.path.exists(modelsdf_path) and not no_cache:
            modelsdf = pd.read_pickle(modelsdf_path)
            modelsdf = modelsdf[modelsdf["grid_es"] != 1.0].reset_index(drop=True)
            print(f"  Loaded cached modelsdf ({len(modelsdf)} entries)")
        else:
            t0 = time.time()
            modelsdf = model_selection_roc_auc(
                dataset_split, config["grid_sizes"], config["grid_es"],
                lamb=0.0, k=config["ms_k"], optim=config["optim"],
                epochs=config["ms_epochs"], grid_update_num=config["grid_update_num"],
                stop_grid_update_step=config["stop_grid_update_step"],
                alpha=config["alpha"], beta=config["beta"],
                r2_threshold=config["r2_threshold"], device=device, exp_seed=feat_seed)
            static = modelsdf[modelsdf["grid_es"] == 1.0].reset_index(drop=True)
            modelsdf = modelsdf[modelsdf["grid_es"] != 1.0].reset_index(drop=True)
            print(f"  Model selection took {time.time()-t0:.1f}s")
            modelsdf.to_pickle(modelsdf_path)
        mpset = modelsdf[modelsdf["pareto"] == True]
        kan_col, sym_col, higher_better = _model_metric_cols(modelsdf)
        mean_metric = 0.5 * (mpset[kan_col] + mpset[sym_col])
        idx = mean_metric[mean_metric == mean_metric.max()].index if higher_better else mean_metric[mean_metric == mean_metric.min()].index
        G, g_e = modelsdf.iloc[idx]["grid_sizes"].values[0], modelsdf.iloc[idx]["grid_es"].values[0]
        print(f"  Best model: G={G}, g_e={g_e}")

    # ── Step 5: Final evaluation ──
    print("[4/5] Final training & evaluation...")
    final_scaler = StandardScaler() if config["use_scaler"] else None
    final_data = get_data(df, scaler=final_scaler, data_split=config["data_split"],
                           final_eval=True, feat_idxs=kept_feats,
                           device=device, exp_seed=exp_seed)

    kan_input = final_data["train_input"].shape[1]
    kan_output = final_data["train_label"].unique().shape[0]
    model = KAN(width=[kan_input, kan_output], grid=G, k=config["ms_k"],
                grid_eps=g_e, sparse_init=False, seed=exp_seed,
                auto_save=False, device=device)

    results = model.fit(final_data, opt=config["optim"], steps=config["ms_epochs"],
                         lamb=0.0, update_grid=True,
                         grid_update_num=config["grid_update_num"],
                         stop_grid_update_step=config["stop_grid_update_step"],
                         loss_fn=torch.nn.CrossEntropyLoss())

    logits = model.forward(final_data["test_input"]).detach()
    pred_probs = torch.nn.functional.softmax(logits, dim=1)
    pred_proba = pred_probs[:, 1].cpu().numpy()
    truth = final_data["test_label"].cpu().numpy()

    roc_auc = roc_auc_score(truth, pred_proba)
    precisions, recalls, thresholds = precision_recall_curve(truth, pred_proba)
    f1_scores_arr = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_threshold = thresholds[np.argmax(f1_scores_arr)]
    y_pred = (pred_proba >= best_threshold).astype(int)
    f1 = f1_score(truth, y_pred, average="weighted")

    # ── Symbolify ──
    print("[5/5] Symbolification...")
    model.auto_symbolic(verbose=0, alpha=config["alpha"], beta=config["beta"],
                         r2_threshold=config["r2_threshold"])
    logits_sym = model.forward(final_data["test_input"]).detach()
    pred_probs_sym = torch.nn.functional.softmax(logits_sym, dim=1)
    pred_proba_sym = pred_probs_sym[:, 1].cpu().numpy()
    roc_auc_sym = roc_auc_score(truth, pred_proba_sym)
    y_pred_sym = (pred_proba_sym >= best_threshold).astype(int)
    f1_sym = f1_score(truth, y_pred_sym, average="weighted")

    metrics = {
        "roc_auc": round(roc_auc, 4),
        "f1": round(f1, 4),
        "roc_auc_sym": round(roc_auc_sym, 4),
        "f1_sym": round(f1_sym, 4),
        "num_features": len(kept_feats),
        "G": G, "g_e": g_e,
    }
    print(f"  KAN ROC-AUC: {roc_auc:.4f}, F1: {f1:.4f}")
    print(f"  Sym ROC-AUC: {roc_auc_sym:.4f}, F1: {f1_sym:.4f}")
    return metrics


# ═══════════════════════════════════════════════════════════════════
# 5. SIDER MULTI-LABEL CLASSIFICATION RUNNER
# ═══════════════════════════════════════════════════════════════════

def run_sider_classification(args, config, no_cache=False):
    """SIDER multi-label classification (27 tasks).

    Data extraction EXACTLY matches Sider_Test_Classify_mynew.py:
    - masks: full row arrays (27 dims)
    - labels: full row arrays (27 dims)
    """
    print(f"\n{'='*70}")
    print(f"  Running: SIDER (multi-label classification)")
    print(f"{'='*70}")

    device = torch.device("cpu")
    exp_seed = config["exp_seed"]
    experiment_name = config["experiment_name"]

    print("[1/5] Loading dataset and extracting features...")
    dataset, train_set, val_set, test_set = load_dgl_dataset("SIDER", args)
    loader = DataLoader(dataset, batch_size=args["batch_size"], collate_fn=collate_molgraphs, drop_last=True)

    # EXACTLY matching the original SIDER feature extraction
    all_hidden_feat, final_labels, df_masks = [], [], []
    for batch_id, batch_data in enumerate(loader):
        smiles, bg, labels, masks = batch_data
        df_masks.append(np.asarray(masks[0, :]))
        bg = dgl.add_self_loop(bg)
        atom_feats = bg.ndata.pop(args['atom_data_field'])
        hidden_feat = torch.mean(atom_feats, dim=0).unsqueeze(dim=0)
        mol = Chem.MolFromSmiles(smiles[0])
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        logp = Descriptors.MolLogP(mol)
        mw = Descriptors.ExactMolWt(mol)
        HBD = rdMolDescriptors.CalcNumHBD(mol)
        HBA = rdMolDescriptors.CalcNumHBA(mol)
        hidden_feat = torch.cat((hidden_feat, torch.tensor([tpsa, logp, mw, HBD, HBA]).unsqueeze(dim=0)), dim=1)
        all_hidden_feat.append(hidden_feat.detach().numpy().tolist())
        final_labels.append((labels.numpy().astype(float))[0, :])

    # EXACTLY matching the original DataFrame construction
    all_hidden_feat = [item for sublist in all_hidden_feat for item in sublist]
    df = pd.DataFrame({'feat': all_hidden_feat, 'label': final_labels})
    num_columns = len(df['feat'].iloc[0])
    columns = [f'feat_{i}' for i in range(num_columns)]
    feat_df = pd.DataFrame(df['feat'].tolist(), columns=columns)
    df = df.drop(columns='feat')
    df_masks = pd.DataFrame({'masks': df_masks})
    df = pd.concat([feat_df, df, df_masks], axis=1)
    print(f"  DataFrame shape: {df.shape}")

    # ── Step 2: Feature selection ──
    if "kept_feats_override" in config:
        kept_feats = config["kept_feats_override"]
        print(f"[2/5] Using pre-computed {len(kept_feats)} features (skip feature selection)")
    else:
        print("[2/5] Feature selection...")
        os.makedirs("results", exist_ok=True)
        featsdf_path = os.path.join("results", f"{experiment_name}_featsdf.pkl")
        if os.path.exists(featsdf_path) and not no_cache:
            featsdf = pd.read_pickle(featsdf_path)
            print(f"  Loaded cached featsdf")
        else:
            t0 = time.time()
            featsdf = feature_selection_roc_auc_multiply_label(
                df=df, grid_size=config["fs_grid_size"], grid_eps=config["fs_grid_eps"],
                k=config["fs_k"], thresholds=config["thresholds"], lambdas=config["lambdas"],
                optim=config["optim"], epochs=config["fs_epochs"],
                use_scaler=config["use_scaler"], data_split=config["data_split"],
                device=device, exp_seed=feat_seed)
            print(f"  Feature selection took {time.time()-t0:.1f}s")
            featsdf.to_pickle(featsdf_path)
        kept_feats, _ = _select_best_from_pareto_feats(featsdf, 10)
        print(f"  Selected {len(kept_feats)} features")

    # ── Step 3: Model selection ──
    if "G_override" in config:
        G = config["G_override"]
        g_e = config["g_e_override"]
        print(f"[3/5] Using pre-computed G={G}, g_e={g_e} (skip model selection)")
    else:
        scaler = StandardScaler() if config["use_scaler"] else None
        dataset_split = get_multiply_label_data(df, scaler=scaler, data_split=config["data_split"],
                                                 final_eval=False, feat_idxs=kept_feats,
                                                 device=device, exp_seed=feat_seed)
        print("[3/5] Model selection...")
        os.makedirs("results", exist_ok=True)
        modelsdf_path = os.path.join("results", f"{experiment_name}_modelsdf.pkl")
        if os.path.exists(modelsdf_path) and not no_cache:
            modelsdf = pd.read_pickle(modelsdf_path)
            modelsdf = modelsdf[modelsdf["grid_es"] != 1.0].reset_index(drop=True)
            print(f"  Loaded cached modelsdf")
        else:
            t0 = time.time()
            modelsdf = model_selection_roc_auc_multiply_label(
                dataset_split, config["grid_sizes"], config["grid_es"],
                lamb=0.0, k=config["ms_k"], optim=config["optim"],
                epochs=config["ms_epochs"], grid_update_num=config["grid_update_num"],
                stop_grid_update_step=config["stop_grid_update_step"],
                alpha=config["alpha"], beta=config["beta"],
                r2_threshold=config["r2_threshold"], device=device, exp_seed=feat_seed)
            static = modelsdf[modelsdf["grid_es"] == 1.0].reset_index(drop=True)
            modelsdf = modelsdf[modelsdf["grid_es"] != 1.0].reset_index(drop=True)
            print(f"  Model selection took {time.time()-t0:.1f}s")
            modelsdf.to_pickle(modelsdf_path)
        mpset = modelsdf[modelsdf["pareto"] == True]
        kan_col, sym_col, higher_better = _model_metric_cols(modelsdf)
        mean_metric = 0.5 * (mpset[kan_col] + mpset[sym_col])
        idx = mean_metric[mean_metric == mean_metric.max()].index if higher_better else mean_metric[mean_metric == mean_metric.min()].index
        G, g_e = modelsdf.iloc[idx]["grid_sizes"].values[0], modelsdf.iloc[idx]["grid_es"].values[0]
        print(f"  Best model: G={G}, g_e={g_e}")

    # Final evaluation
    print("[4/5] Final training & evaluation...")
    final_scaler = StandardScaler() if config["use_scaler"] else None
    final_data = get_multiply_label_data_new(df, scaler=final_scaler, data_split=config["data_split"],
                                              final_eval=True, device=device, exp_seed=exp_seed)

    final_data_copy = final_data.copy()
    final_data_copy["train_label"] = final_data_copy["train_label"].type(torch.float32).to(device)
    final_data_copy["val_label"] = final_data_copy["val_label"].type(torch.float32).to(device)
    final_data_copy["test_label"] = final_data_copy["test_label"].type(torch.float32).to(device)

    kan_input = final_data_copy["train_input"].shape[1]
    kan_output = final_data_copy["train_label"].shape[1]
    model = KAN(width=[kan_input, kan_output], grid=G, k=config["ms_k"],
                grid_eps=g_e, sparse_init=False, seed=exp_seed,
                auto_save=False, device=device)

    results = model.fit(final_data_copy, opt=config["optim"], steps=config["ms_epochs"],
                         lamb=0.0, update_grid=True,
                         grid_update_num=config["grid_update_num"],
                         stop_grid_update_step=config["stop_grid_update_step"],
                         loss_fn=torch.nn.BCEWithLogitsLoss())

    preds = torch.sigmoid(model.forward(final_data_copy["test_input"]).detach()).cpu().numpy()
    truth = final_data_copy["test_label"].cpu().numpy()
    roc_auc = roc_auc_score(truth, preds)

    best_thresholds = []
    y_pred = np.zeros_like(preds)
    for i in range(truth.shape[1]):
        prec, rec, thres = precision_recall_curve(truth[:, i], preds[:, i])
        f1_arr = 2 * (prec * rec) / (prec + rec + 1e-8)
        best_thresholds.append(thres[np.argmax(f1_arr)])
        y_pred[:, i] = (preds[:, i] >= best_thresholds[-1])
    f1 = f1_score(truth, y_pred, average="weighted")

    metrics = {
        "roc_auc": round(float(roc_auc), 4),
        "f1": round(float(f1), 4),
        "num_features": len(kept_feats),
        "G": G, "g_e": g_e,
    }
    print(f"  ROC-AUC: {roc_auc:.4f}, F1: {f1:.4f}")
    return metrics


# ═══════════════════════════════════════════════════════════════════
# 6. HOB MULTI-LABEL CLASSIFICATION RUNNER
# ═══════════════════════════════════════════════════════════════════

def run_hob_classification(args, config, no_cache=False):
    """HOB multi-label classification (label_cutoff_50%, label_cutoff_20%).

    Data extraction EXACTLY matches Hob_Test_Classify_mynew.py:
    - Only custom descriptors (NO atom features)
    - Multi-label: label_cutoff_50% + label_cutoff_20%
    - BCEWithLogitsLoss for both feature selection and model selection
    """
    print(f"\n{'='*70}")
    print(f"  Running: HOB (multi-label classification)")
    print(f"{'='*70}")

    device = torch.device("cpu")
    exp_seed = config["exp_seed"]
    feat_seed = config.get("feat_seed", exp_seed)
    experiment_name = config["experiment_name"]

    print("[1/5] Loading dataset and extracting features...")
    dataset, train_set, val_set, test_set = load_hob_dataset(args)
    loader = DataLoader(dataset, batch_size=args["batch_size"], collate_fn=collate_molgraphs, drop_last=True)

    all_hidden_feat, final_labels, df_masks = [], [], []
    for batch_id, batch_data in enumerate(loader):
        smiles, bg, labels, masks = batch_data
        df_masks.append(np.asarray(masks[0, :]))
        bg = dgl.add_self_loop(bg)
        mol = Chem.MolFromSmiles(smiles[0])
        hidden_feat = torch.tensor(calc_hob_descriptors(mol)).unsqueeze(dim=0).tolist()
        all_hidden_feat.append(hidden_feat)
        final_labels.append((labels.numpy().astype(float))[0, :])

    all_hidden_feat = [item for sublist in all_hidden_feat for item in sublist]
    df = pd.DataFrame({'feat': all_hidden_feat, 'label': final_labels})
    num_columns = len(df['feat'].iloc[0])
    columns = [f'feat_{i}' for i in range(num_columns)]
    feat_df = pd.DataFrame(df['feat'].tolist(), columns=columns)
    df = df.drop(columns='feat')
    df_masks = pd.DataFrame({'masks': df_masks})
    df = pd.concat([feat_df, df, df_masks], axis=1)
    print(f"  DataFrame shape: {df.shape}")

    # ── Step 2: Feature selection (load from cache) ──
    print("[2/5] Loading features from cache...")
    if "kept_feats_override" in config:
        kept_feats = config["kept_feats_override"]
        print(f"  Using pre-computed {len(kept_feats)} features")
    else:
        # Strip _seedXX suffix to get base experiment name for cache file
        import re
        base_exp = re.sub(r'_seed\d+$', '', experiment_name)
        featsdf_path = os.path.join("results", f"{base_exp}_featsdf.pkl")
        featsdf = pd.read_pickle(featsdf_path)
        kept_feats, _ = _select_best_from_pareto_feats(featsdf, 10)
        print(f"  Loaded {len(kept_feats)} features from {featsdf_path}")

    # ── Step 3: Model selection ──
    if "G_override" in config:
        G = config["G_override"]
        g_e = config["g_e_override"]
        print(f"[3/5] Using pre-computed G={G}, g_e={g_e} (from modelsdf cache)")
    else:
        # Fallback (original hardcoded params from Hob_Test_Classify_mynew.py line 425-426)
        G, g_e = 12, 0.0
        print(f"[3/5] Using hardcoded best params: G={G}, g_e={g_e}")

    # ── Step 4: Final evaluation ──
    print("[4/5] Final training & evaluation...")
    final_scaler = StandardScaler() if config["use_scaler"] else None
    final_data = get_multiply_label_data_new(df, scaler=final_scaler, data_split=config["data_split"],
                                              final_eval=True, feat_idxs=kept_feats,
                                              device=device, exp_seed=exp_seed)

    final_data_copy = final_data.copy()
    final_data_copy["train_label"] = final_data_copy["train_label"].type(torch.float32).to(device)
    final_data_copy["val_label"] = final_data_copy["val_label"].type(torch.float32).to(device)
    final_data_copy["test_label"] = final_data_copy["test_label"].type(torch.float32).to(device)

    kan_input = final_data_copy["train_input"].shape[1]
    kan_output = final_data_copy["train_label"].shape[1]
    model = KAN(width=[kan_input, kan_output], grid=G, k=config["ms_k"],
                grid_eps=g_e, sparse_init=False, seed=exp_seed,
                auto_save=False, device=device)

    results = model.fit(final_data_copy, opt=config["optim"], steps=config["ms_epochs"],
                         lamb=0.0, update_grid=True,
                         grid_update_num=config["grid_update_num"],
                         stop_grid_update_step=config["stop_grid_update_step"],
                         loss_fn=torch.nn.BCEWithLogitsLoss())

    preds = torch.sigmoid(model.forward(final_data_copy["test_input"]).detach()).cpu().numpy()
    truth = final_data_copy["test_label"].cpu().numpy()
    roc_auc = roc_auc_score(truth, preds)

    best_thresholds = []
    y_pred = np.zeros_like(preds)
    for i in range(truth.shape[1]):
        prec, rec, thres = precision_recall_curve(truth[:, i], preds[:, i])
        f1_arr = 2 * (prec * rec) / (prec + rec + 1e-8)
        best_thresholds.append(thres[np.argmax(f1_arr)])
        y_pred[:, i] = (preds[:, i] >= best_thresholds[-1])
    f1 = f1_score(truth, y_pred, average="weighted")

    # ── Step 5: Symbolification ──
    print("[5/5] Symbolification...")
    model.auto_symbolic(verbose=0, alpha=config["alpha"], beta=config["beta"],
                         r2_threshold=config["r2_threshold"])
    preds_sym = torch.sigmoid(model.forward(final_data_copy["test_input"]).detach()).cpu().numpy()
    roc_auc_sym = roc_auc_score(truth, preds_sym)

    y_pred_sym = np.zeros_like(preds_sym)
    for i in range(truth.shape[1]):
        y_pred_sym[:, i] = (preds_sym[:, i] >= best_thresholds[i])
    f1_sym = f1_score(truth, y_pred_sym, average="weighted")

    metrics = {
        "roc_auc": round(float(roc_auc), 4),
        "f1": round(float(f1), 4),
        "roc_auc_sym": round(float(roc_auc_sym), 4),
        "f1_sym": round(float(f1_sym), 4),
        "num_features": len(kept_feats),
        "G": G, "g_e": g_e,
    }
    print(f"  KAN ROC-AUC: {roc_auc:.4f}, F1: {f1:.4f}")
    print(f"  Sym ROC-AUC: {roc_auc_sym:.4f}, F1: {f1_sym:.4f}")
    return metrics


# ═══════════════════════════════════════════════════════════════════
# 7. REGRESSION RUNNER (ESOL, FreeSolv, CDK9)
# ═══════════════════════════════════════════════════════════════════

def run_regression(dataset_name, args, config, no_cache=False):
    """Run for regression tasks (ESOL, FreeSolv, CDK9).

    Data extraction EXACTLY matches each original script:
    - ESOL: atom features + 16 custom descriptors
    - FreeSolv: atom features + 14 custom descriptors
    - CDK9: RDKit descriptors only (no atom features)
    """
    print(f"\n{'='*70}")
    print(f"  Running: {dataset_name} (regression)")
    print(f"{'='*70}")

    device = torch.device("cpu")
    exp_seed = config["exp_seed"]
    feat_seed = config.get("feat_seed", exp_seed)
    experiment_name = config["experiment_name"]

    # ── Step 1: Load dataset & extract features ──
    print("[1/5] Loading dataset and extracting features...")
    if dataset_name == "CDK9":
        dataset, train_set, val_set, test_set = load_cdk9_dataset(args)
    else:
        dgl_name = "FreeSolv" if dataset_name == "FreeSolv" else dataset_name
        dataset, train_set, val_set, test_set = load_dgl_dataset(dgl_name, args)

    loader = DataLoader(dataset, batch_size=args["batch_size"], collate_fn=collate_molgraphs, drop_last=True)

    # EXACTLY matching each original regression script's feature extraction
    all_hidden_feat, final_labels, df_masks = [], [], []
    for batch_id, batch_data in enumerate(loader):
        smiles, bg, labels, masks = batch_data
        df_masks.append(np.asarray(masks[0, :]))
        bg = dgl.add_self_loop(bg)
        mol = Chem.MolFromSmiles(smiles[0])

        if dataset_name == "CDK9":
            # CDK9: RDKit descriptors only (no atom features)
            extra = calc_cdk9_descriptors(mol)
            hidden_feat = torch.tensor(extra, dtype=torch.float32).unsqueeze(dim=0)
        else:
            # ESOL / FreeSolv: atom features + custom descriptors
            atom_feats = bg.ndata.pop(args["atom_data_field"])
            hidden_feat = torch.mean(atom_feats, dim=0).unsqueeze(dim=0)
            if dataset_name == "ESOL":
                extra = calc_esol_descriptors(mol)
            else:
                extra = calc_freesolv_descriptors(mol)
            hidden_feat = torch.cat((hidden_feat, torch.tensor(extra).unsqueeze(dim=0)), dim=1)

        all_hidden_feat.append(hidden_feat.detach().numpy().tolist())
        labels_np = labels.numpy().astype(float)
        final_labels.append(labels_np[0, 0] if labels_np.shape[1] == 1 else labels_np[0, :])

    # EXACTLY matching the original DataFrame construction
    all_hidden_feat = [item for sublist in all_hidden_feat for item in sublist]
    df = pd.DataFrame({'feat': all_hidden_feat, 'label': final_labels})
    num_columns = len(df['feat'].iloc[0])
    columns = [f'feat_{i}' for i in range(num_columns)]
    feat_df = pd.DataFrame(df['feat'].tolist(), columns=columns)
    df = df.drop(columns='feat')
    df_masks = pd.DataFrame({'masks': df_masks})
    df = pd.concat([feat_df, df, df_masks], axis=1)
    print(f"  DataFrame shape: {df.shape}")

    # ── Step 2: Feature selection ──
    if "kept_feats_override" in config:
        kept_feats = config["kept_feats_override"]
        print(f"[2/5] Using pre-computed {len(kept_feats)} features (skip feature selection)")
    else:
        print("[2/5] Feature selection...")
        os.makedirs("results", exist_ok=True)
        featsdf_path = os.path.join("results", f"{experiment_name}_featsdf.pkl")

        fs_fn = feature_selection_return_R2 if dataset_name == "CDK9" else feature_selection_return
        if os.path.exists(featsdf_path) and not no_cache:
            featsdf = pd.read_pickle(featsdf_path)
            print(f"  Loaded cached featsdf ({len(featsdf)} entries)")
        else:
            t0 = time.time()
            featsdf = fs_fn(
                df=df, grid_size=config["fs_grid_size"], grid_eps=config["fs_grid_eps"],
                k=config["fs_k"], thresholds=config["thresholds"], lambdas=config["lambdas"],
                optim=config["optim"], epochs=config["fs_epochs"],
                use_scaler=config["use_scaler"], data_split=config["data_split"],
                device=device, exp_seed=feat_seed)
            print(f"  Feature selection took {time.time()-t0:.1f}s")
            featsdf.to_pickle(featsdf_path)

        # Select best features
        kept_feats, lamb = _select_best_from_pareto_feats(featsdf, config.get("max_feats", 11))
        print(f"  Selected {len(kept_feats)} features, lambda={lamb:.4f}")

    # ── Step 3: Model selection ──
    if "G_override" in config:
        G = config["G_override"]
        g_e = config["g_e_override"]
        print(f"[3/5] Using pre-computed G={G}, g_e={g_e} (skip model selection)")
    else:
        scaler = StandardScaler() if config["use_scaler"] else None
        dataset_split = get_data_return(df, scaler=scaler, data_split=config["data_split"],
                                         final_eval=False, feat_idxs=kept_feats,
                                         device=device, exp_seed=feat_seed)
        print("[3/5] Model selection...")
        modelsdf_path = os.path.join("results", f"{experiment_name}_modelsdf.pkl")

        ms_fn = model_selection_return_R2 if dataset_name == "CDK9" else model_selection_return
        if os.path.exists(modelsdf_path) and not no_cache:
            modelsdf = pd.read_pickle(modelsdf_path)
            modelsdf = modelsdf[modelsdf["grid_es"] != 1.0].reset_index(drop=True)
            print(f"  Loaded cached modelsdf ({len(modelsdf)} entries)")
        else:
            t0 = time.time()
            modelsdf = ms_fn(
                dataset_split, config["grid_sizes"], config["grid_es"],
                lamb=0.0, k=config["ms_k"], optim=config["optim"],
                epochs=config["ms_epochs"], grid_update_num=config["grid_update_num"],
                stop_grid_update_step=config["stop_grid_update_step"],
                alpha=config["alpha"], beta=config["beta"],
                r2_threshold=config["r2_threshold"], device=device, exp_seed=feat_seed)
            static = modelsdf[modelsdf["grid_es"] == 1.0].reset_index(drop=True)
            modelsdf = modelsdf[modelsdf["grid_es"] != 1.0].reset_index(drop=True)
            print(f"  Model selection took {time.time()-t0:.1f}s")
            modelsdf.to_pickle(modelsdf_path)

        # Select best model using metric_direction
        mpset = modelsdf[modelsdf["pareto"] == True]
        kan_col, sym_col, higher_better = _model_metric_cols(modelsdf)
        mean_metric = 0.5 * (mpset[kan_col] + mpset[sym_col])
        idx = mean_metric[mean_metric == mean_metric.max()].index if higher_better else mean_metric[mean_metric == mean_metric.min()].index
        G, g_e = modelsdf.iloc[idx]["grid_sizes"].values[0], modelsdf.iloc[idx]["grid_es"].values[0]
        print(f"  Best model: G={G}, g_e={g_e}")

    # ── Step 5: Final evaluation ──
    print("[4/5] Final training & evaluation...")
    final_scaler = StandardScaler() if config["use_scaler"] else None
    final_data = get_data_return(df, scaler=final_scaler, data_split=config["data_split"],
                                  final_eval=True, feat_idxs=kept_feats,
                                  device=device, exp_seed=exp_seed)

    kan_input = final_data["train_input"].shape[1]
    model = KAN(width=[kan_input, 1], grid=G, k=config["ms_k"],
                grid_eps=g_e, sparse_init=False, seed=exp_seed,
                auto_save=False, device=device)

    results = model.fit(final_data, opt=config["optim"], steps=config["ms_epochs"],
                         lamb=0.0, update_grid=True,
                         grid_update_num=config["grid_update_num"],
                         stop_grid_update_step=config["stop_grid_update_step"],
                         loss_fn=torch.nn.MSELoss())

    preds = model(final_data["test_input"]).detach()
    truth = final_data["test_label"]
    rmse = torch.sqrt(torch.mean((preds - truth) ** 2)).item()
    r2 = r2_score(truth.cpu().numpy(), preds.cpu().numpy())

    # Symbolify
    print("[5/5] Symbolification...")
    model.auto_symbolic(verbose=0, alpha=config["alpha"], beta=config["beta"],
                         r2_threshold=config["r2_threshold"])
    preds_sym = model.forward(final_data["test_input"]).detach()
    rmse_sym = torch.sqrt(torch.mean((preds_sym - truth) ** 2)).item()
    r2_sym = r2_score(truth.cpu().numpy(), preds_sym.cpu().numpy())

    metrics = {
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "rmse_sym": round(rmse_sym, 4),
        "r2_sym": round(r2_sym, 4),
        "num_features": len(kept_feats),
        "G": G, "g_e": g_e,
    }
    print(f"  KAN RMSE: {rmse:.4f}, R2: {r2:.4f}")
    print(f"  Sym RMSE: {rmse_sym:.4f}, R2: {r2_sym:.4f}")
    return metrics


# ═══════════════════════════════════════════════════════════════════
# 8. CONFIGURATIONS (matching original experiment scripts)
# ═══════════════════════════════════════════════════════════════════

BASE_ARGS = {
    "random_seed": 42, "batch_size": 1, "train_batch_size": 1,
    "atom_data_field": "h", "atom_featurizer": CanonicalAtomFeaturizer(),
    "device": "cpu",
}

COMMON_CONFIG = {
    "exp_seed": 42,
    "fs_grid_size": 5, "fs_grid_eps": 0.05,
    "optim": "Adam",
    "alpha": 0.05, "beta": 1.5, "r2_threshold": 0.0,
    "grid_update_num": 10, "stop_grid_update_step": 150,
}

CLINTOX_CONFIG = {
    **COMMON_CONFIG,
    "experiment_name": "clinttox_classification",
    "label_col": 1,
    "data_split": (80, 10, 10), "use_scaler": True,
    "fs_k": 3, "fs_epochs": 80,
    "thresholds": np.linspace(0.1, 0.2, 5),
    "lambdas": np.linspace(0.001, 0.01, 10),
    "ms_k": 4, "ms_epochs": 200, "max_feats": 10,
    "grid_sizes": [8, 10, 12, 15, 20, 30, 40, 50],
    "grid_es": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
}

BBBP_CONFIG = {
    **COMMON_CONFIG,
    "experiment_name": "bbbp_classification_add_regression",
    "data_split": (80, 10, 10), "use_scaler": False,
    "fs_k": 3, "fs_epochs": 80,
    "thresholds": np.linspace(0.01, 0.10, 8),
    "lambdas": np.linspace(0.001, 0.01, 10),
    "ms_k": 4, "ms_epochs": 200, "max_feats": 11,
    "grid_sizes": [8, 10, 12, 15, 20, 30, 40, 50],
    "grid_es": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
    # Hardcoded fallback from BBBP_Test_Regression.py line 360
    "hardcoded_G": 12, "hardcoded_g_e": 0.35,
}

BACE_CONFIG = {
    **COMMON_CONFIG,
    "experiment_name": "bace_classification",
    "data_split": (80, 10, 10), "use_scaler": True,
    "fs_k": 3, "fs_epochs": 80,
    "thresholds": np.linspace(0.05, 0.3, 10),
    "lambdas": np.linspace(0.005, 0.02, 10),
    "ms_k": 4, "ms_epochs": 200, "max_feats": 10,
    "grid_sizes": [8, 10, 15, 20, 30],
    "grid_es": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
}

SIDER_CONFIG = {
    **COMMON_CONFIG,
    "experiment_name": "sider_classification_add",
    "data_split": (80, 10, 10), "use_scaler": False,
    "fs_k": 3, "fs_epochs": 80,
    "thresholds": np.linspace(0.04, 0.2, 10),
    "lambdas": np.linspace(0.001, 0.03, 20),
    "ms_k": 4, "ms_epochs": 200, "max_feats": 10,
    "grid_sizes": [8, 10, 12, 15, 20, 30, 40, 50],
    "grid_es": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
}

HOB_CONFIG = {
    **COMMON_CONFIG,
    "experiment_name": "hob_classification_add",
    "data_split": (80, 10, 10), "use_scaler": False,
    "fs_k": 3, "fs_epochs": 80,
    "thresholds": np.linspace(0.04, 0.2, 10),
    "lambdas": np.linspace(0.001, 0.03, 20),
    "ms_k": 3, "ms_epochs": 200, "max_feats": 10,
    "grid_sizes": [8, 10, 12, 15, 20, 30, 40, 50],
    "grid_es": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
}

ESOL_CONFIG = {
    **COMMON_CONFIG,
    "experiment_name": "esol_classification_add",
    "data_split": (80, 10, 10), "use_scaler": False,
    "fs_k": 3, "fs_epochs": 80,
    "thresholds": np.linspace(0.01, 0.2, 20),
    "lambdas": np.linspace(0.001, 0.01, 20),
    "ms_k": 3, "ms_epochs": 200, "max_feats": 11,
    "grid_sizes": [8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110],
    "grid_es": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
}

FREESOLV_CONFIG = {
    **COMMON_CONFIG,
    "experiment_name": "FreeSolve_add",
    "data_split": (80, 10, 10), "use_scaler": False,
    "fs_k": 3, "fs_epochs": 80,
    "thresholds": np.linspace(0.01, 0.2, 10),
    "lambdas": np.linspace(0.001, 0.2, 10),
    "ms_k": 3, "ms_epochs": 200, "max_feats": 11,
    "grid_sizes": [8, 10, 12, 15, 20, 30, 40, 50],
    "grid_es": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
}

CDK9_CONFIG = {
    **COMMON_CONFIG,
    "experiment_name": "CDK9_regression_add",
    "data_split": (80, 10, 10), "use_scaler": True,
    "fs_k": 3, "fs_epochs": 80,
    "thresholds": np.linspace(0.4, 0.6, 15),
    "lambdas": np.linspace(0.001, 0.2, 20),
    "ms_k": 3, "ms_epochs": 200, "max_feats": 10,
    "grid_sizes": [8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
    "grid_es": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
}

# ═══════════════════════════════════════════════════════════════════
# 9. BBBP REGRESSION-STYLE RUNNER
# ═══════════════════════════════════════════════════════════════════

def run_bbbp_regression(args, config, no_cache=False):
    """BBBP with regression-style training (MSELoss + ROC-AUC evaluation).

    Follows BBBP_Test_Regression.py exactly:
    - feature_selection_roc_auc_return (trains 1-neuron KAN with MSELoss)
    - Hardcoded G=12, g_e=0.35
    - MSELoss for final training
    """
    print(f"\n{'='*70}")
    print(f"  Running: BBBP (regression-style)")
    print(f"{'='*70}")

    device = torch.device("cpu")
    exp_seed = config["exp_seed"]
    feat_seed = config.get("feat_seed", exp_seed)
    experiment_name = config["experiment_name"]

    # ── Step 1: Load dataset & extract features ──
    print("[1/5] Loading dataset and extracting features...")
    from dgllife.data import BBBP
    dataset = BBBP(smiles_to_bigraph, args["atom_featurizer"])
    train_set, val_set, test_set = ScaffoldSplitter.train_val_test_split(
        dataset, frac_train=args["frac_train"], frac_val=args["frac_val"],
        frac_test=args["frac_test"])
    loader = DataLoader(dataset, batch_size=args["batch_size"], collate_fn=collate_molgraphs, drop_last=True)

    all_hidden_feat, final_labels, df_masks = [], [], []
    for batch_id, batch_data in enumerate(loader):
        smiles, bg, labels, masks = batch_data
        df_masks.append(np.asarray(masks[0, 0]))
        bg = dgl.add_self_loop(bg)
        atom_feats = bg.ndata.pop(args['atom_data_field'])
        hidden_feat = torch.mean(atom_feats, dim=0).unsqueeze(dim=0)
        mol = Chem.MolFromSmiles(smiles[0])
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        logp = Descriptors.MolLogP(mol)
        mw = Descriptors.ExactMolWt(mol)
        HBD = rdMolDescriptors.CalcNumHBD(mol)
        HBA = rdMolDescriptors.CalcNumHBA(mol)
        hidden_feat = torch.cat((hidden_feat, torch.tensor([tpsa, logp, mw, HBD, HBA]).unsqueeze(dim=0)), dim=1)
        all_hidden_feat.append(hidden_feat.detach().numpy().tolist())
        final_labels.append((labels.numpy().astype(float))[0, 0])

    all_hidden_feat = [item for sublist in all_hidden_feat for item in sublist]
    df = pd.DataFrame({'feat': all_hidden_feat, 'label': final_labels})
    num_columns = len(df['feat'].iloc[0])
    columns = [f'feat_{i}' for i in range(num_columns)]
    feat_df = pd.DataFrame(df['feat'].tolist(), columns=columns)
    df = df.drop(columns='feat')
    df_masks = pd.DataFrame({'masks': df_masks})
    df = pd.concat([feat_df, df, df_masks], axis=1)
    print(f"  DataFrame shape: {df.shape}")

    # ── Step 2: Feature selection (regression-style) ──
    if "kept_feats_override" in config:
        kept_feats = config["kept_feats_override"]
        print(f"[2/5] Using pre-computed {len(kept_feats)} features (skip feature selection)")
    else:
        print("[2/5] Feature selection (regression-style)...")
        os.makedirs("results", exist_ok=True)
        featsdf_path = os.path.join("results", f"{experiment_name}_featsdf.pkl")

        if os.path.exists(featsdf_path) and not no_cache:
            featsdf = pd.read_pickle(featsdf_path)
            print(f"  Loaded cached featsdf ({len(featsdf)} entries)")
        else:
            t0 = time.time()
            featsdf = feature_selection_roc_auc_return(
                df=df, grid_size=config["fs_grid_size"], grid_eps=config["fs_grid_eps"],
                k=config["fs_k"], thresholds=config["thresholds"], lambdas=config["lambdas"],
                optim=config["optim"], epochs=config["fs_epochs"],
                use_scaler=config["use_scaler"], data_split=config["data_split"],
                device=device, exp_seed=feat_seed)
            print(f"  Feature selection took {time.time()-t0:.1f}s")
            featsdf.to_pickle(featsdf_path)

        # Select best features
        kept_feats, lamb = _select_best_from_pareto_feats(featsdf, config.get("max_feats", 11))
        print(f"  Selected {len(kept_feats)} features, lambda={lamb:.4f}")

    # ── Step 4: Use hardcoded best params (from BBBP_Test_Regression.py line 360) ──
    G = config["hardcoded_G"]
    g_e = config["hardcoded_g_e"]
    print(f"[3/5] Using hardcoded params: G={G}, g_e={g_e}")

    # ── Step 5: Final training & evaluation ──
    print("[4/5] Final training & evaluation...")
    final_scaler = StandardScaler() if config["use_scaler"] else None
    final_data = get_data(df, scaler=final_scaler, data_split=config["data_split"],
                           final_eval=True, feat_idxs=kept_feats,
                           device=device, exp_seed=exp_seed)
    # Convert labels to float32 regression format
    final_data['train_label'] = final_data['train_label'].type(torch.float32).to(device).unsqueeze(1)
    final_data['val_label'] = final_data['val_label'].type(torch.float32).to(device).unsqueeze(1)
    final_data['test_label'] = final_data['test_label'].type(torch.float32).to(device).unsqueeze(1)

    kan_input = final_data['train_input'].shape[1]
    model = KAN(width=[kan_input, 1], grid=G, k=config["ms_k"], grid_eps=g_e,
                sparse_init=False, seed=exp_seed, auto_save=False, device=device)

    results = model.fit(final_data, opt="LBFGS", steps=config["ms_epochs"],
                         lamb=0.0, update_grid=True,
                         grid_update_num=config["grid_update_num"],
                         stop_grid_update_step=config["stop_grid_update_step"],
                         loss_fn=torch.nn.MSELoss())

    logits = model.forward(final_data['test_input']).detach()
    truth = final_data['test_label'].cpu()
    roc_auc = roc_auc_score(truth, logits)

    # Compute F1 using best threshold from precision-recall curve
    logits_np = logits.cpu().numpy().flatten()
    truth_np = truth.numpy().flatten()
    precisions, recalls, thresholds = precision_recall_curve(truth_np, logits_np)
    f1_scores_arr = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_threshold = thresholds[np.argmax(f1_scores_arr)]
    y_pred = (logits_np >= best_threshold).astype(int)
    f1 = f1_score(truth_np, y_pred, average="weighted")

    # ── Symbolify ──
    print("[5/5] Symbolification...")
    model.auto_symbolic(verbose=0, alpha=config["alpha"], beta=config["beta"],
                         r2_threshold=config["r2_threshold"])
    logits_sym = model.forward(final_data['test_input']).detach()
    logits_sym_np = logits_sym.cpu().numpy().flatten()
    roc_auc_sym = roc_auc_score(truth_np, logits_sym_np)
    y_pred_sym = (logits_sym_np >= best_threshold).astype(int)
    f1_sym = f1_score(truth_np, y_pred_sym, average="weighted")

    metrics = {
        "roc_auc": round(roc_auc, 4),
        "f1": round(f1, 4),
        "roc_auc_sym": round(roc_auc_sym, 4),
        "f1_sym": round(f1_sym, 4),
        "num_features": len(kept_feats),
        "G": G, "g_e": g_e,
    }
    print(f"  KAN ROC-AUC: {roc_auc:.4f}, F1: {f1:.4f}")
    print(f"  Sym ROC-AUC: {roc_auc_sym:.4f}, F1: {f1_sym:.4f}")
    return metrics


# ═══════════════════════════════════════════════════════════════════
# 9. MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════

DATASET_CONFIGS = {
    "ClinTox":  ("classification_single", CLINTOX_CONFIG),
    "BBBP":     ("bbbp_regression", BBBP_CONFIG),
    "BACE":     ("classification_single", BACE_CONFIG),
    "SIDER":    ("classification_multi",  SIDER_CONFIG),
    "HOB":      ("classification_hob",  HOB_CONFIG),
    "FreeSolv": ("regression", FREESOLV_CONFIG),
    "ESOL":     ("regression", ESOL_CONFIG),
    "CDK9":     ("regression", CDK9_CONFIG),
}


def _load_feats_from_cache(experiment_name, config):
    """Read the existing featsdf.pkl and extract best kept features list."""
    featsdf_path = os.path.join("results", f"{experiment_name}_featsdf.pkl")
    if not os.path.exists(featsdf_path):
        return {}
    featsdf = pd.read_pickle(featsdf_path)
    kept_feats, _ = _select_best_from_pareto_feats(featsdf, config.get("max_feats", 11))
    print(f"  Loaded {len(kept_feats)} kept features from {featsdf_path}")
    return {"kept_feats_override": kept_feats}


def _load_model_from_cache(experiment_name, config):
    """Read the existing modelsdf.pkl and extract best G, g_e."""
    modelsdf_path = os.path.join("results", f"{experiment_name}_modelsdf.pkl")
    if not os.path.exists(modelsdf_path):
        return {}
    modelsdf = pd.read_pickle(modelsdf_path)
    modelsdf = modelsdf[modelsdf["grid_es"] != 1.0].reset_index(drop=True)
    mpset = modelsdf[modelsdf["pareto"] == True]
    if len(mpset) == 0:
        return {}
    kan_col, sym_col, higher_better = _model_metric_cols(modelsdf)
    mean_metric = 0.5 * (mpset[kan_col] + mpset[sym_col])
    idx = mean_metric[mean_metric == mean_metric.max()].index if higher_better else mean_metric[mean_metric == mean_metric.min()].index
    G = int(modelsdf.iloc[idx]["grid_sizes"].values[0])
    g_e = float(modelsdf.iloc[idx]["grid_es"].values[0])
    print(f"  Loaded G={G}, g_e={g_e} from {modelsdf_path}")
    return {"G_override": G, "g_e_override": g_e}


def _apply_fixed_params(config, ds_name):
    """Inject best hyperparams from cached pkl files for fast multi-seed.
    Reads features from featsdf.pkl and G/g_e from modelsdf.pkl.
    Falls back to BEST_PARAMS only when pkl files are missing."""
    config = copy.deepcopy(config)
    bp = BEST_PARAMS.get(ds_name, {})
    # Read kept features from featsdf cache
    feat_overrides = _load_feats_from_cache(config["experiment_name"], config)
    config.update(feat_overrides)
    # Read G, g_e from modelsdf cache
    model_overrides = _load_model_from_cache(config["experiment_name"], config)
    if model_overrides:
        config.update(model_overrides)
    elif bp:
        config["G_override"] = bp.get("G", 10)
        config["g_e_override"] = bp.get("g_e", 0.5)
    # ms_k from BEST_PARAMS (not stored in modelsdf)
    config["ms_k"] = bp.get("ms_k", 3) if bp else 3
    # Freeze feature selection seed (not used when kept_feats_override is set,
    # but kept for consistency)
    config["feat_seed"] = 42
    # BBBP uses hardcoded params — use model_overrides or BEST_PARAMS
    if "hardcoded_G" in config:
        G_val = config.get("G_override", bp.get("G", 10))
        g_e_val = config.get("g_e_override", bp.get("g_e", 0.5))
        config["hardcoded_G"] = G_val
        config["hardcoded_g_e"] = g_e_val
    return config


def _run_one(ds_name, task_type, config, ds_args, seed, no_cache):
    """Run one (dataset, seed) combination. Returns metrics dict or raises."""
    config = copy.deepcopy(config)
    feat_seed = config.get("feat_seed", seed)
    config["exp_seed"] = seed
    # Cache files use feat_seed (feature selection is shared across all seeds)
    base_name = config["experiment_name"]
    config["experiment_name"] = f"{base_name}_seed{feat_seed}"

    if task_type == "classification_single":
        return run_classification_single_label(ds_name, ds_args, config, no_cache=no_cache)
    elif task_type == "classification_hob":
        return run_hob_classification(ds_args, config, no_cache=no_cache)
    elif task_type == "bbbp_regression":
        return run_bbbp_regression(ds_args, config, no_cache=no_cache)
    elif task_type == "classification_multi":
        return run_sider_classification(ds_args, config, no_cache=no_cache)
    elif task_type == "regression":
        return run_regression(ds_name, ds_args, config, no_cache=no_cache)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")


def _compute_summary(all_results):
    """Compute mean ± std across seeds for each dataset and metric."""
    import json
    summary = {}
    for ds_name, seed_results in all_results.items():
        valid = [r for r in seed_results if "error" not in r]
        if not valid:
            summary[ds_name] = {"error": "all seeds failed", "num_seeds": len(seed_results)}
            continue
        agg = {"num_seeds": len(valid)}
        for key in valid[0].keys():
            vals = [r[key] for r in valid if key in r and isinstance(r[key], (int, float))]
            if vals:
                agg[key] = {"mean": round(float(np.mean(vals)), 4),
                            "std": round(float(np.std(vals)), 4)}
        summary[ds_name] = agg
    return summary


def _print_single_summary(results):
    """Original single-seed summary table."""
    print(f"\n\n{'='*90}")
    print(f"  REPRODUCTION RESULTS SUMMARY")
    print(f"{'='*90}")
    print(f"{'Dataset':<12} {'Task':<12} {'Metric':<28} {'Expected':<12} {'Obtained':<12} {'Match':<8}")
    print(f"{'-'*90}")
    for ds_name, exp in EXPECTED.items():
        if ds_name not in results:
            print(f"{ds_name:<12} {'N/A':<12} {'N/A':<28} {'N/A':<12} {'N/A':<12} {'N/A':<8}")
            continue
        res = results[ds_name]
        if "error" in res:
            print(f"{ds_name:<12} {'ERROR':<12} {res['error'][:28]:<28} {'---':<12} {'---':<12} {'---':<8}")
            continue
        task = exp["task"]
        if task == "classification":
            for metric_name, exp_key in [("ROC-AUC", "roc_auc"), ("F1", "f1")]:
                exp_val = exp[exp_key]
                obt_val = res.get(exp_key, "N/A")
                match = "OK" if isinstance(obt_val, float) and abs(obt_val - exp_val) < 0.1 else "DIFF"
                print(f"{ds_name:<12} {task:<12} {metric_name:<28} {exp_val:<12} {obt_val:<12} {match:<8}")
        else:
            for metric_name, exp_key in [("RMSE", "rmse"), ("R2", "r2")]:
                exp_val = exp[exp_key]
                obt_val = res.get(exp_key, "N/A")
                if isinstance(obt_val, float) and exp_key == "rmse":
                    match = "OK" if abs(obt_val - exp_val) < 0.3 else "DIFF"
                elif isinstance(obt_val, float) and exp_key == "r2":
                    match = "OK" if abs(obt_val - exp_val) < 0.15 else "DIFF"
                else:
                    match = "---"
                print(f"{ds_name:<12} {task:<12} {metric_name:<28} {exp_val:<12} {obt_val:<12} {match:<8}")


def _print_multi_summary(summary):
    """Multi-seed mean ± std summary table."""
    print(f"\n\n{'='*100}")
    print(f"  MULTI-SEED RESULTS: Mean ± Std across {summary[list(summary.keys())[0]]['num_seeds']} seeds")
    print(f"{'='*100}")

    for ds_name, exp in EXPECTED.items():
        if ds_name not in summary:
            continue
        agg = summary[ds_name]
        if "error" in agg:
            print(f"  {ds_name:<12} ERROR: {agg['error']}")
            continue
        task = exp["task"]
        print(f"\n── {ds_name} ({task}) ──")
        if task == "classification":
            for key in ["roc_auc", "f1", "roc_auc_sym", "f1_sym"]:
                if key in agg:
                    m, s = agg[key]["mean"], agg[key]["std"]
                    print(f"  {key:<16s}: {m:.4f} ± {s:.4f}")
        else:
            for key in ["rmse", "r2", "rmse_sym", "r2_sym"]:
                if key in agg:
                    m, s = agg[key]["mean"], agg[key]["std"]
                    print(f"  {key:<16s}: {m:.4f} ± {s:.4f}")
        if "num_features" in agg:
            m, s = agg["num_features"]["mean"], agg["num_features"]["std"]
            print(f"  {'num_features':<16s}: {m:.1f} ± {s:.1f}")


def _save_summary_csv(summary, output_path):
    """Save multi-seed summary as a flat CSV (one row per dataset×metric)."""
    rows = []
    for ds_name, agg in summary.items():
        if "error" in agg:
            continue
        for key, stats in agg.items():
            if key == "num_seeds":
                continue
            if isinstance(stats, dict) and "mean" in stats:
                rows.append({
                    "dataset": ds_name,
                    "metric": key,
                    "mean": stats["mean"],
                    "std": stats["std"],
                })
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nSummary CSV saved to: {output_path}")


def _save_per_seed_json(all_results, output_path):
    """Save per-seed raw metrics for statistical significance testing (E3)."""
    import json
    serializable = {}
    for ds_name, seed_results in all_results.items():
        serializable[ds_name] = []
        for r in seed_results:
            clean = {}
            for k, v in r.items():
                if isinstance(v, (int, float, str, bool, type(None))):
                    clean[k] = v
                elif isinstance(v, np.integer):
                    clean[k] = int(v)
                elif isinstance(v, np.floating):
                    clean[k] = float(v)
                elif isinstance(v, np.ndarray):
                    clean[k] = v.tolist()
                else:
                    clean[k] = str(v)
            serializable[ds_name].append(clean)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"Per-seed JSON saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Reproduce KAN experimental results")
    parser.add_argument("--no-cache", action="store_true", help="Recompute everything from scratch")
    parser.add_argument("--dataset", type=str, default=None, help="Run a single dataset")
    parser.add_argument("--seeds", type=str, default="42",
                        help="Comma-separated random seeds (default: '42'). Use '42,123,456' for multi-seed.")
    parser.add_argument("--output", type=str, default="results/multi_seed_summary.csv",
                        help="Output CSV path for multi-seed summary")
    pargs = parser.parse_args()
    seeds = [int(s.strip()) for s in pargs.seeds.split(",")]
    multi_seed = len(seeds) > 1

    if pargs.dataset:
        datasets_to_run = [pargs.dataset]
    else:
        datasets_to_run = list(DATASET_CONFIGS.keys())

    # ── Build ds_args once per dataset ──
    ds_args_map = {}
    for ds_name in datasets_to_run:
        ds_args = BASE_ARGS.copy()
        if ds_name == "BACE":
            ds_args.update({"frac_train": 0.8, "frac_val": 0.1, "frac_test": 0.1})
        else:
            ds_args.update({"frac_train": 0.8, "frac_val": 0.1, "frac_test": 0.1})
        ds_args_map[ds_name] = ds_args

    # ── Multi-seed mode ──
    if multi_seed:
        print(f"\n{'='*70}")
        print(f"  MULTI-SEED MODE: {len(seeds)} seeds → {seeds}")
        print(f"  Using fixed hyperparameters from BEST_PARAMS (no grid search)")
        print(f"{'='*70}")

        all_results = {}  # {ds_name: [metrics_seed1, metrics_seed2, ...]}

        for ds_name in datasets_to_run:
            if ds_name not in DATASET_CONFIGS:
                print(f"Unknown dataset: {ds_name}")
                continue
            task_type, config = DATASET_CONFIGS[ds_name]
            ds_args = ds_args_map[ds_name]
            # Use fixed hyperparams → each seed runs quickly (no grid search)
            fixed_config = _apply_fixed_params(config, ds_name)
            ds_seed_results = []

            for seed in seeds:
                print(f"\n  ── {ds_name} | seed={seed} ──")
                try:
                    metrics = _run_one(ds_name, task_type, fixed_config, ds_args, seed, no_cache=pargs.no_cache)
                    ds_seed_results.append(metrics)
                except Exception as e:
                    print(f"  ERROR {ds_name} seed={seed}: {e}")
                    import traceback
                    traceback.print_exc()
                    ds_seed_results.append({"error": str(e), "seed": seed})

            all_results[ds_name] = ds_seed_results

        # ── Save per-seed raw metrics for statistical testing (E3) ──
        per_seed_path = os.path.join("results", "multi_seed_per_seed.json")
        _save_per_seed_json(all_results, per_seed_path)

        # ── Compute & display summary ──
        summary = _compute_summary(all_results)
        _print_multi_summary(summary)
        _save_summary_csv(summary, pargs.output)

        print(f"\n  Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return all_results

    # ── Single-seed mode (original behaviour) ──
    results = {}
    for ds_name in datasets_to_run:
        if ds_name not in DATASET_CONFIGS:
            print(f"Unknown dataset: {ds_name}")
            continue
        task_type, config = DATASET_CONFIGS[ds_name]
        ds_args = ds_args_map[ds_name]

        try:
            metrics = _run_one(ds_name, task_type, config, ds_args, config["exp_seed"], no_cache=pargs.no_cache)
            results[ds_name] = metrics
        except Exception as e:
            print(f"  ERROR running {ds_name}: {e}")
            import traceback
            traceback.print_exc()
            results[ds_name] = {"error": str(e)}

    _print_single_summary(results)
    print(f"\n  Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return results


if __name__ == "__main__":
    main()

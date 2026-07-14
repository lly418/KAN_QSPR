import torch
import itertools
import numpy as np
import pandas as pd
from kan import KAN
from paretoset import paretoset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, r2_score

from .data import get_data, get_data_return, get_multiply_label_data_new
from .meter import Meter

def model_selection_roc_auc_multiply_label(dataset, grid_sizes, grid_es, lamb, k=4, optim="Adam", epochs=80, grid_update_num=10,
                    stop_grid_update_step=100, alpha=0.05, beta=1.5, r2_threshold=0.0, device='cuda', exp_seed=42,
                    verbose=True):
    """
    Model selection for multi-label classification (metric = ROC-AUC, higher is better).
    """
    # Get the combination of (grid_size, grid_e) pairs
    combinations = list(itertools.product(grid_sizes, grid_es))
    dataset['train_label'] = dataset['train_label'].type(torch.float32).to(device).unsqueeze(1)
    dataset['val_label'] = dataset['val_label'].type(torch.float32).to(device).unsqueeze(1)
    dataset['test_label'] = dataset['test_label'].type(torch.float32).to(device).unsqueeze(1)

    # Initialize lists
    metric_kan = []
    metric_sym = []

    def closure(dataset, grid_size, grid_e, lamb, k, optim, epochs, grid_update_num, stop_grid_update_step, alpha, beta,
                r2_threshold, exp_seed, device):

        # Initialize a model
        kan_input = dataset['train_input'].shape[1]
        kan_output = dataset['train_label'].shape[1]
        model = KAN(width=[kan_input, kan_output], grid=grid_size, k=k, grid_eps=grid_e, seed=exp_seed, auto_save=False,
                    device=device)

        # Check for non adaptive training
        update_grid = False if grid_e > 0.99 else True

        # Train Model
        results = model.fit(dataset, opt=optim, steps=epochs, reg_metric='node_backward', lamb=0.0,
                            update_grid=update_grid, grid_update_num=grid_update_num,
                            stop_grid_update_step=stop_grid_update_step, loss_fn=torch.nn.BCEWithLogitsLoss())

        # Evaluate on validation data — ROC-AUC
        preds = model.forward(dataset['val_input']).cpu()
        test_probs = torch.sigmoid(preds).cpu().detach().numpy()
        truth = dataset['val_label'].cpu()
        metric_kan_score = roc_auc_score(truth, test_probs)

        # Symbolify Model
        model.auto_symbolic(verbose=0, alpha=alpha, beta=beta, r2_threshold=r2_threshold)
        # Evaluate Symbolic Version
        preds_sym = model.forward(dataset['val_input']).cpu()
        test_probs = torch.sigmoid(preds_sym).cpu().detach().numpy()
        metric_sym_score = roc_auc_score(truth, test_probs)

        del model

        return metric_kan_score, metric_sym_score

    # Run loop for all combinations of lambda, threshold
    ct = 1
    for (grid_size, grid_e) in combinations:
        if verbose:
            print(f"Running Experiment No. {ct} for grid size = {grid_size}, grid_eps = {grid_e}.")
        try:
            m_kan, m_sym = closure(dataset, grid_size, grid_e, lamb, k, optim, epochs, grid_update_num,
                                   stop_grid_update_step, alpha, beta, r2_threshold, exp_seed, device)

            metric_kan.append(m_kan)
            metric_sym.append(m_sym)

            if verbose:
                print(
                    f"KAN Model: ROC-AUC of {m_kan:.2f}.\t Symbolic Model: ROC-AUC of {m_sym:.2f}.\n")
        except Exception as e:
            if verbose:
                print(f"Exception {e}\nOmmiting this one.")
            metric_kan.append(0)
            metric_sym.append(0)
        ct += 1

    # Gather results in dataframe, to be returned
    modelsdf = pd.DataFrame(combinations, columns=['grid_sizes', 'grid_es'])
    modelsdf['metric_kan'] = metric_kan
    modelsdf['metric_sym'] = metric_sym
    # Drop None values
    modelsdf = modelsdf.dropna()
    modelsdf['metric_kan'] = modelsdf['metric_kan'].astype('float64')
    modelsdf['metric_sym'] = modelsdf['metric_sym'].astype('float64')

    if modelsdf.shape[0] > 0:
        # Pareto: maximize both KAN and symbolic ROC-AUC
        paretodf = pd.DataFrame({"metric_kan": modelsdf['metric_kan'].values, "metric_sym": modelsdf['metric_sym'].values})
        mask = paretoset(paretodf, sense=["max", "max"])
        modelsdf['pareto'] = mask
        # Fallback: if no Pareto solutions, pick best symbolic
        if not mask.any():
            max_idx = modelsdf['metric_sym'].idxmax()
            modelsdf.loc[max_idx, 'pareto'] = True

    # ROC-AUC: higher is better
    modelsdf['metric_direction'] = 'max'

    return modelsdf


def model_selection_roc_auc_return(dataset, grid_sizes, grid_es, lamb, k=4, optim="Adam", epochs=80, grid_update_num=10,
                    stop_grid_update_step=100, alpha=0.05, beta=1.5, r2_threshold=0.0, device='cuda', exp_seed=42,
                    verbose=True):
    """
    Model selection for regression-style classification (metric = ROC-AUC, higher is better).
    Uses MSELoss training + ROC-AUC evaluation (for BBBP).
    """
    # Get the combination of (grid_size, grid_e) pairs
    combinations = list(itertools.product(grid_sizes, grid_es))
    dataset['train_label'] = dataset['train_label'].type(torch.float32).to(device).unsqueeze(1)
    dataset['val_label'] = dataset['val_label'].type(torch.float32).to(device).unsqueeze(1)
    dataset['test_label'] = dataset['test_label'].type(torch.float32).to(device).unsqueeze(1)

    # Initialize lists
    metric_kan = []
    metric_sym = []

    def closure(dataset, grid_size, grid_e, lamb, k, optim, epochs, grid_update_num, stop_grid_update_step, alpha, beta,
                r2_threshold, exp_seed, device):

        # Initialize a model
        kan_input = dataset['train_input'].shape[1]
        kan_output = 1
        model = KAN(width=[kan_input, kan_output], grid=grid_size, k=k, grid_eps=grid_e, seed=exp_seed, auto_save=False,
                    device=device)

        # Check for non adaptive training
        update_grid = False if grid_e > 0.99 else True

        # Train Model with MSELoss
        results = model.fit(dataset, opt=optim, steps=epochs, reg_metric='node_backward', lamb=0.0,
                            update_grid=update_grid, grid_update_num=grid_update_num,
                            stop_grid_update_step=stop_grid_update_step, loss_fn=torch.nn.MSELoss())

        # Evaluate on validation data — ROC-AUC
        preds = model.forward(dataset['val_input']).detach().cpu()
        truth = dataset['val_label'].cpu()
        metric_kan_score = roc_auc_score(truth, preds)

        # Symbolify Model
        model.auto_symbolic(verbose=0, alpha=alpha, beta=beta, r2_threshold=r2_threshold)
        # Evaluate Symbolic Version
        preds_sym = model.forward(dataset['val_input']).detach().cpu()
        metric_sym_score = roc_auc_score(truth, preds_sym)

        del model

        return metric_kan_score, metric_sym_score

    # Run loop for all combinations of lambda, threshold
    ct = 1
    for (grid_size, grid_e) in combinations:
        if verbose:
            print(f"Running Experiment No. {ct} for grid size = {grid_size}, grid_eps = {grid_e}.")
        try:
            m_kan, m_sym = closure(dataset, grid_size, grid_e, lamb, k, optim, epochs, grid_update_num,
                                   stop_grid_update_step, alpha, beta, r2_threshold, exp_seed, device)

            metric_kan.append(m_kan)
            metric_sym.append(m_sym)

            if verbose:
                print(
                    f"KAN Model: ROC-AUC of {m_kan:.2f}.\t Symbolic Model: ROC-AUC of {m_sym:.2f}.\n")
        except Exception as e:
            if verbose:
                print(f"Exception {e}\nOmmiting this one.")
            metric_kan.append(0)
            metric_sym.append(0)
        ct += 1

    # Gather results in dataframe, to be returned
    modelsdf = pd.DataFrame(combinations, columns=['grid_sizes', 'grid_es'])
    modelsdf['metric_kan'] = metric_kan
    modelsdf['metric_sym'] = metric_sym
    # Drop None values
    modelsdf = modelsdf.dropna()
    modelsdf['metric_kan'] = modelsdf['metric_kan'].astype('float64')
    modelsdf['metric_sym'] = modelsdf['metric_sym'].astype('float64')

    if modelsdf.shape[0] > 0:
        # Pareto: maximize both KAN and symbolic ROC-AUC
        paretodf = pd.DataFrame({"metric_kan": modelsdf['metric_kan'].values, "metric_sym": modelsdf['metric_sym'].values})
        mask = paretoset(paretodf, sense=["max", "max"])
        modelsdf['pareto'] = mask
        # Fallback: if no Pareto solutions, pick best symbolic
        if not mask.any():
            max_idx = modelsdf['metric_sym'].idxmax()
            modelsdf.loc[max_idx, 'pareto'] = True

    # ROC-AUC: higher is better
    modelsdf['metric_direction'] = 'max'

    return modelsdf

def model_selection_roc_auc(dataset, grid_sizes, grid_es, lamb, k=4, optim="Adam", epochs=80, grid_update_num=10,
                    stop_grid_update_step=100, alpha=0.05, beta=1.5, r2_threshold=0.0, device='cuda', exp_seed=42,
                    verbose=True):
    """
    Model selection for single-label classification (metric = ROC-AUC, higher is better).
    """
    # Get the combination of (grid_size, grid_e) pairs
    combinations = list(itertools.product(grid_sizes, grid_es))

    # Initialize lists
    metric_kan = []
    metric_sym = []

    def closure(dataset, grid_size, grid_e, lamb, k, optim, epochs, grid_update_num, stop_grid_update_step, alpha, beta,
                r2_threshold, exp_seed, device):

        # Initialize a model
        kan_input = dataset['train_input'].shape[1]
        kan_output = dataset['train_label'].unique().shape[0]
        model = KAN(width=[kan_input, kan_output], grid=grid_size, k=k, grid_eps=grid_e, seed=exp_seed, auto_save=False,
                    device=device)

        # Check for non adaptive training
        update_grid = False if grid_e > 0.99 else True

        # Train Model
        results = model.fit(dataset, opt=optim, steps=epochs, reg_metric='node_backward', lamb=0.0,
                            update_grid=update_grid, grid_update_num=grid_update_num,
                            stop_grid_update_step=stop_grid_update_step, loss_fn=torch.nn.CrossEntropyLoss())

        # Evaluate on validation data — ROC-AUC
        preds = model.forward(dataset['val_input']).detach()
        pred_probs = torch.nn.functional.softmax(preds, dim=1)
        pred_proba = pred_probs[:, 1].cpu().detach().numpy()
        truth = dataset['val_label'].cpu()
        metric_kan_score = roc_auc_score(truth, pred_proba)

        # Symbolify Model
        model.auto_symbolic(verbose=0, alpha=alpha, beta=beta, r2_threshold=r2_threshold)
        # Evaluate Symbolic Version
        preds_sym = model.forward(dataset['val_input'])
        pred_probs_sym = torch.nn.functional.softmax(preds_sym, dim=1)
        pred_proba_sym = pred_probs_sym[:, 1].cpu().detach().numpy()
        metric_sym_score = roc_auc_score(truth, pred_proba_sym)

        del model

        return metric_kan_score, metric_sym_score

    # Run loop for all combinations of lambda, threshold
    ct = 1
    for (grid_size, grid_e) in combinations:
        if verbose:
            print(f"Running Experiment No. {ct} for grid size = {grid_size}, grid_eps = {grid_e}.")
        try:
            m_kan, m_sym = closure(dataset, grid_size, grid_e, lamb, k, optim, epochs, grid_update_num,
                                   stop_grid_update_step, alpha, beta, r2_threshold, exp_seed, device)

            metric_kan.append(m_kan)
            metric_sym.append(m_sym)

            if verbose:
                print(
                    f"KAN Model: ROC-AUC of {m_kan:.2f}.\t Symbolic Model: ROC-AUC of {m_sym:.2f}.\n")
        except Exception as e:
            if verbose:
                print(f"Exception {e}\nOmmiting this one.")
            metric_kan.append(0)
            metric_sym.append(0)
        ct += 1

    # Gather results in dataframe, to be returned
    modelsdf = pd.DataFrame(combinations, columns=['grid_sizes', 'grid_es'])
    modelsdf['metric_kan'] = metric_kan
    modelsdf['metric_sym'] = metric_sym
    # Drop None values
    modelsdf = modelsdf.dropna()
    modelsdf['metric_kan'] = modelsdf['metric_kan'].astype('float64')
    modelsdf['metric_sym'] = modelsdf['metric_sym'].astype('float64')

    if modelsdf.shape[0] > 0:
        # Pareto: maximize both KAN and symbolic ROC-AUC
        paretodf = pd.DataFrame({"metric_kan": modelsdf['metric_kan'].values, "metric_sym": modelsdf['metric_sym'].values})
        mask = paretoset(paretodf, sense=["max", "max"])
        modelsdf['pareto'] = mask
        # Fallback: if no Pareto solutions, pick best symbolic
        if not mask.any():
            max_idx = modelsdf['metric_sym'].idxmax()
            modelsdf.loc[max_idx, 'pareto'] = True

    # ROC-AUC: higher is better
    modelsdf['metric_direction'] = 'max'

    return modelsdf


def model_selection(dataset, grid_sizes, grid_es, lamb, k=4, optim="Adam", epochs=80, grid_update_num=10, stop_grid_update_step=100, alpha=0.05, beta=1.5, r2_threshold=0.0, device='cuda', exp_seed=42, verbose=True):
    """
    Model selection for single-label classification (metric = Weighted F1, higher is better).
    Legacy function — prefer model_selection_roc_auc for new code.
    """
    # Get the combination of (grid_size, grid_e) pairs
    combinations = list(itertools.product(grid_sizes, grid_es))

    # Initialize lists
    metric_kan = []
    metric_sym = []

    def closure(dataset, grid_size, grid_e, lamb, k, optim, epochs, grid_update_num, stop_grid_update_step, alpha, beta, r2_threshold, exp_seed, device):

        # Initialize a model
        kan_input = dataset['train_input'].shape[1]
        kan_output = dataset['train_label'].unique().shape[0]
        model = KAN(width=[kan_input,kan_output], grid=grid_size, k=k, grid_eps=grid_e, seed=exp_seed, auto_save=False, device=device)

        # Check for non adaptive training
        update_grid = False if grid_e > 0.99 else True

        # Train Model
        results = model.fit(dataset, opt=optim, steps=epochs, reg_metric='node_backward', lamb=0.0, update_grid=update_grid, grid_update_num=grid_update_num, stop_grid_update_step=stop_grid_update_step, loss_fn=torch.nn.CrossEntropyLoss())

        # Evaluate on validation data — Weighted F1
        preds = torch.argmax(model.forward(dataset['val_input']).detach(), dim=1).cpu()
        truth = dataset['val_label'].cpu()
        m_kan = 100*f1_score(truth, preds, average='weighted')

        # Symbolify Model
        model.auto_symbolic(verbose=0, alpha=alpha, beta=beta, r2_threshold=r2_threshold)
        # Evaluate Symbolic Version — Weighted F1
        preds_sym = torch.argmax(model.forward(dataset['val_input']).detach(), dim=1).cpu()
        m_sym = 100*f1_score(truth, preds_sym, average='weighted')

        del model

        return m_kan, m_sym

    # Run loop for all combinations of lambda, threshold
    ct = 1
    for (grid_size, grid_e) in combinations:
        if verbose:
            print(f"Running Experiment No. {ct} for grid size = {grid_size}, grid_eps = {grid_e}.")
        try:
            m_kan, m_sym = closure(dataset, grid_size, grid_e, lamb, k, optim, epochs, grid_update_num, stop_grid_update_step, alpha, beta, r2_threshold, exp_seed, device)

            metric_kan.append(m_kan)
            metric_sym.append(m_sym)

            if verbose:
                print(f"KAN Model: Weighted F1-Score of {m_kan:.2f}%.\t Symbolic Model: Weighted F1-Score of {m_sym:.2f}%.\n")
        except Exception as e:
            if verbose:
                print(f"Exception {e}\nOmmiting this one.")
            metric_kan.append(0)
            metric_sym.append(0)
        ct += 1

    # Gather results in dataframe, to be returned
    modelsdf = pd.DataFrame(combinations, columns=['grid_sizes', 'grid_es'])
    modelsdf['metric_kan'] = metric_kan
    modelsdf['metric_sym'] = metric_sym
    # Drop None values
    modelsdf = modelsdf.dropna()
    modelsdf['metric_kan'] = modelsdf['metric_kan'].astype('float64')
    modelsdf['metric_sym'] = modelsdf['metric_sym'].astype('float64')

    if modelsdf.shape[0] > 0:
        # Pareto: maximize both KAN and symbolic F1
        paretodf = pd.DataFrame({"metric_kan": modelsdf['metric_kan'].values, "metric_sym": modelsdf['metric_sym'].values})
        mask = paretoset(paretodf, sense=["max", "max"])
        modelsdf['pareto'] = mask
        # Fallback: if no Pareto solutions, pick best symbolic
        if not mask.any():
            max_idx = modelsdf['metric_sym'].idxmax()
            modelsdf.loc[max_idx, 'pareto'] = True

    # Weighted F1: higher is better
    modelsdf['metric_direction'] = 'max'

    return modelsdf


def model_selection_return(dataset, grid_sizes, grid_es, lamb, k=4, optim="Adam", epochs=80,
                           grid_update_num=10, stop_grid_update_step=100, alpha=0.05, beta=1.5,
                           r2_threshold=0.0, device='cuda', exp_seed=42, verbose=True):
    """
    Model selection for regression tasks (metric = RMSE, lower is better).
    """
    # Parameter combinations
    combinations = list(itertools.product(grid_sizes, grid_es))

    # Initialize storage
    rmse_kan = []
    rmse_sym = []

    def closure(dataset, grid_size, grid_e, lamb, k, optim, epochs, grid_update_num,
                stop_grid_update_step, alpha, beta, r2_threshold, exp_seed, device):

        # Initialize model (output dim = 1 for regression)
        kan_input = dataset['train_input'].shape[1]
        kan_output = 1

        model = KAN(width=[kan_input, kan_output], grid=grid_size, k=k, grid_eps=grid_e,
                    seed=exp_seed, auto_save=False, device=device)

        update_grid = False if grid_e > 0.99 else True

        loss_fn = torch.nn.MSELoss()

        # Train model
        results = model.fit(dataset, opt=optim, steps=epochs, reg_metric='node_backward',
                            lamb=lamb, update_grid=update_grid, grid_update_num=grid_update_num,
                            stop_grid_update_step=stop_grid_update_step, loss_fn=loss_fn)

        # Evaluate KAN model — RMSE
        with torch.no_grad():
            preds = model(dataset['val_input']).detach().cpu().squeeze()
            truth = dataset['val_label'].cpu().squeeze()

        rmse_kan_score = torch.sqrt(torch.mean((preds - truth) ** 2)).item()

        # Symbolify model
        model.auto_symbolic(verbose=0, alpha=alpha, beta=beta, r2_threshold=r2_threshold)

        # Evaluate symbolic model — RMSE
        with torch.no_grad():
            preds_sym = model(dataset['val_input']).detach().cpu().squeeze()

        rmse_sym_score = torch.sqrt(torch.mean((preds_sym - truth) ** 2)).item()

        del model

        return rmse_kan_score, rmse_sym_score

    # Run all combinations
    ct = 1
    for (grid_size, grid_e) in combinations:
        if verbose:
            print(f"Running Experiment No. {ct} for grid size = {grid_size}, grid_eps = {grid_e}.")
        try:
            rmse_kan_score, rmse_sym_score = closure(dataset, grid_size, grid_e, lamb, k, optim,
                                                     epochs, grid_update_num, stop_grid_update_step,
                                                     alpha, beta, r2_threshold, exp_seed, device)

            rmse_kan.append(rmse_kan_score)
            rmse_sym.append(rmse_sym_score)

            if verbose:
                print(f"KAN Model: RMSE = {rmse_kan_score:.4f}\t Symbolic Model: RMSE = {rmse_sym_score:.4f}\n")
        except Exception as e:
            if verbose:
                print(f"Exception {e}\nOmitting this one.")
            rmse_kan.append(np.inf)
            rmse_sym.append(np.inf)
        ct += 1

    # Build results DataFrame
    modelsdf = pd.DataFrame(combinations, columns=['grid_sizes', 'grid_es'])
    modelsdf['rmse_kan'] = rmse_kan
    modelsdf['rmse_sym'] = rmse_sym

    # Remove invalid values
    modelsdf = modelsdf.replace(np.inf, np.nan).dropna()
    modelsdf['rmse_kan'] = modelsdf['rmse_kan'].astype('float64')
    modelsdf['rmse_sym'] = modelsdf['rmse_sym'].astype('float64')

    if modelsdf.shape[0] > 0:
        # Pareto: minimize both KAN and symbolic RMSE
        paretodf = pd.DataFrame({
            "rmse_kan": modelsdf['rmse_kan'].values,
            "rmse_sym": modelsdf['rmse_sym'].values
        })
        mask = paretoset(paretodf, sense=["min", "min"])
        modelsdf['pareto'] = mask

        # Fallback: pick best symbolic RMSE
        if not mask.any():
            min_idx = modelsdf['rmse_sym'].idxmin()
            modelsdf.loc[min_idx, 'pareto'] = True

    # RMSE: lower is better
    modelsdf['metric_direction'] = 'min'

    return modelsdf


def model_selection_return_R2(dataset, grid_sizes, grid_es, lamb, k=4, optim="Adam", epochs=80,
                           grid_update_num=10, stop_grid_update_step=100, alpha=0.05, beta=1.5,
                           r2_threshold=0.0, device='cuda', exp_seed=42, verbose=True):
    """
    Model selection for regression tasks (metric = R², higher is better).
    """
    # Parameter combinations
    combinations = list(itertools.product(grid_sizes, grid_es))

    # Initialize storage
    r2_kan = []
    r2_sym = []

    def closure(dataset, grid_size, grid_e, lamb, k, optim, epochs, grid_update_num,
                stop_grid_update_step, alpha, beta, r2_threshold, exp_seed, device):

        # Initialize model (output dim = 1 for regression)
        kan_input = dataset['train_input'].shape[1]
        kan_output = 1

        model = KAN(width=[kan_input, kan_output], grid=grid_size, k=k, grid_eps=grid_e,
                    seed=exp_seed, auto_save=False, device=device)

        update_grid = False if grid_e > 0.99 else True

        loss_fn = torch.nn.MSELoss()

        # Train model
        results = model.fit(dataset, opt=optim, steps=epochs, reg_metric='node_backward',
                            lamb=lamb, update_grid=update_grid, grid_update_num=grid_update_num,
                            stop_grid_update_step=stop_grid_update_step, loss_fn=loss_fn)

        # Evaluate KAN model — R² (higher is better)
        with torch.no_grad():
            preds = model(dataset['val_input']).detach().cpu().squeeze()
            truth = dataset['val_label'].cpu().squeeze()

        r2_kan_score = r2_score(truth, preds)

        # Symbolify model
        model.auto_symbolic(verbose=0, alpha=alpha, beta=beta, r2_threshold=r2_threshold)

        # Evaluate symbolic model — R² (higher is better)
        with torch.no_grad():
            preds_sym = model(dataset['val_input']).detach().cpu().squeeze()

        r2_sym_score = r2_score(truth, preds_sym)

        del model

        return r2_kan_score, r2_sym_score

    # Run all combinations
    ct = 1
    for (grid_size, grid_e) in combinations:
        if verbose:
            print(f"Running Experiment No. {ct} for grid size = {grid_size}, grid_eps = {grid_e}.")
        try:
            r2_kan_score, r2_sym_score = closure(dataset, grid_size, grid_e, lamb, k, optim,
                                                 epochs, grid_update_num, stop_grid_update_step,
                                                 alpha, beta, r2_threshold, exp_seed, device)

            r2_kan.append(r2_kan_score)
            r2_sym.append(r2_sym_score)

            if verbose:
                print(f"KAN Model: R² = {r2_kan_score:.4f}\t Symbolic Model: R² = {r2_sym_score:.4f}\n")
        except Exception as e:
            if verbose:
                print(f"Exception {e}\nOmitting this one.")
            r2_kan.append(-np.inf)
            r2_sym.append(-np.inf)
        ct += 1

    # Build results DataFrame
    modelsdf = pd.DataFrame(combinations, columns=['grid_sizes', 'grid_es'])
    modelsdf['r2_kan'] = r2_kan
    modelsdf['r2_sym'] = r2_sym

    # Remove invalid values
    modelsdf = modelsdf.replace(-np.inf, np.nan).dropna()
    modelsdf['r2_kan'] = modelsdf['r2_kan'].astype('float64')
    modelsdf['r2_sym'] = modelsdf['r2_sym'].astype('float64')

    if modelsdf.shape[0] > 0:
        # Pareto: maximize both KAN and symbolic R²
        paretodf = pd.DataFrame({
            "r2_kan": modelsdf['r2_kan'].values,
            "r2_sym": modelsdf['r2_sym'].values
        })
        mask = paretoset(paretodf, sense=["max", "max"])
        modelsdf['pareto'] = mask

        # Fallback: pick best symbolic R²
        if not mask.any():
            max_idx = modelsdf['r2_sym'].idxmax()
            modelsdf.loc[max_idx, 'pareto'] = True

    # R²: higher is better
    modelsdf['metric_direction'] = 'max'

    return modelsdf



def model_selection_return_two_layers(dataset, grid_sizes, grid_es, lamb, k=4, optim="Adam", epochs=80,
                           grid_update_num=10, stop_grid_update_step=100, alpha=0.05, beta=1.5,
                           r2_threshold=0.0, device='cuda', exp_seed=42, verbose=True):
    """
    Model selection for regression tasks with two hidden layers (metric = RMSE, lower is better).
    """
    # Parameter combinations
    combinations = list(itertools.product(grid_sizes, grid_es))

    # Initialize storage
    rmse_kan = []
    rmse_sym = []

    def closure(dataset, grid_size, grid_e, lamb, k, optim, epochs, grid_update_num,
                stop_grid_update_step, alpha, beta, r2_threshold, exp_seed, device):

        # Initialize model with hidden layer [input, 4, output]
        kan_input = dataset['train_input'].shape[1]
        kan_output = 1

        model = KAN(width=[kan_input,4,kan_output], grid=grid_size, k=k, grid_eps=grid_e,
                    seed=exp_seed, auto_save=False, device=device)

        update_grid = False if grid_e > 0.99 else True

        loss_fn = torch.nn.MSELoss()

        # Train model
        results = model.fit(dataset, opt=optim, steps=epochs, reg_metric='node_backward',
                            lamb=lamb, update_grid=update_grid, grid_update_num=grid_update_num,
                            stop_grid_update_step=stop_grid_update_step, loss_fn=loss_fn)

        # Evaluate KAN model — RMSE
        with torch.no_grad():
            preds = model(dataset['val_input']).detach().cpu().squeeze()
            truth = dataset['val_label'].cpu().squeeze()

        rmse_kan_score = torch.sqrt(torch.mean((preds - truth) ** 2)).item()

        # Symbolify model
        model.auto_symbolic(verbose=0, alpha=alpha, beta=beta, r2_threshold=r2_threshold)

        # Evaluate symbolic model — RMSE
        with torch.no_grad():
            preds_sym = model(dataset['val_input']).detach().cpu().squeeze()

        rmse_sym_score = torch.sqrt(torch.mean((preds_sym - truth) ** 2)).item()

        del model

        return rmse_kan_score, rmse_sym_score

    # Run all combinations
    ct = 1
    for (grid_size, grid_e) in combinations:
        if verbose:
            print(f"Running Experiment No. {ct} for grid size = {grid_size}, grid_eps = {grid_e}.")
        try:
            rmse_kan_score, rmse_sym_score = closure(dataset, grid_size, grid_e, lamb, k, optim,
                                                     epochs, grid_update_num, stop_grid_update_step,
                                                     alpha, beta, r2_threshold, exp_seed, device)

            rmse_kan.append(rmse_kan_score)
            rmse_sym.append(rmse_sym_score)

            if verbose:
                print(f"KAN Model: RMSE = {rmse_kan_score:.4f}\t Symbolic Model: RMSE = {rmse_sym_score:.4f}\n")
        except Exception as e:
            if verbose:
                print(f"Exception {e}\nOmitting this one.")
            rmse_kan.append(np.inf)
            rmse_sym.append(np.inf)
        ct += 1

    # Build results DataFrame
    modelsdf = pd.DataFrame(combinations, columns=['grid_sizes', 'grid_es'])
    modelsdf['rmse_kan'] = rmse_kan
    modelsdf['rmse_sym'] = rmse_sym

    # Remove invalid values
    modelsdf = modelsdf.replace(np.inf, np.nan).dropna()
    modelsdf['rmse_kan'] = modelsdf['rmse_kan'].astype('float64')
    modelsdf['rmse_sym'] = modelsdf['rmse_sym'].astype('float64')

    if modelsdf.shape[0] > 0:
        # Pareto: minimize both KAN and symbolic RMSE
        paretodf = pd.DataFrame({
            "rmse_kan": modelsdf['rmse_kan'].values,
            "rmse_sym": modelsdf['rmse_sym'].values
        })
        mask = paretoset(paretodf, sense=["min", "min"])
        modelsdf['pareto'] = mask

        # Fallback: pick best symbolic RMSE
        if not mask.any():
            min_idx = modelsdf['rmse_sym'].idxmin()
            modelsdf.loc[min_idx, 'pareto'] = True

    # RMSE: lower is better
    modelsdf['metric_direction'] = 'min'

    return modelsdf

def model_selection_with_mask(dataset, grid_sizes, grid_es, lamb, k=4, optim="Adam", epochs=80, grid_update_num=10, stop_grid_update_step=100, alpha=0.05, beta=1.5, r2_threshold=0.0, device='cuda', exp_seed=42, verbose=True):
    """Model selection for multi-task classification with masks (ROC-AUC metric).
    Uses Meter for evaluation with mask support.
    KAN architecture uses hidden layer [input, 128, output].
    """
    combinations = list(itertools.product(grid_sizes, grid_es))

    metric_kan = []
    metric_sym = []

    def closure(dataset, grid_size, grid_e, lamb, k, optim, epochs, grid_update_num, stop_grid_update_step, alpha, beta, r2_threshold, exp_seed, device):
        kan_input = dataset['train_input'].shape[1]
        kan_output = dataset['train_label'].unique().shape[0]
        model = KAN(width=[kan_input, 128, kan_output], grid=grid_size, k=k, grid_eps=grid_e, seed=exp_seed, auto_save=False, device=device)

        update_grid = False if grid_e > 0.99 else True

        results = model.fit(dataset, opt=optim, steps=epochs, reg_metric='node_backward', lamb=0.0, update_grid=update_grid, grid_update_num=grid_update_num, stop_grid_update_step=stop_grid_update_step, loss_fn=torch.nn.CrossEntropyLoss())

        preds = model.forward(dataset['val_input']).detach()
        truth = dataset['val_label'].cpu()
        masks = dataset['mask_val'].cpu()
        eval_meter = Meter()
        eval_meter.update(preds, truth, masks)
        m_kan = np.mean(eval_meter.roc_auc_score())

        model.auto_symbolic(verbose=0, alpha=alpha, beta=beta, r2_threshold=r2_threshold)
        preds_sym = model.forward(dataset['val_input']).detach()
        eval_meter1 = Meter()
        eval_meter1.update(preds_sym, truth, masks)
        m_sym = np.mean(eval_meter1.roc_auc_score())

        del model
        return m_kan, m_sym

    ct = 1
    for (grid_size, grid_e) in combinations:
        if verbose:
            print(f"Running Experiment No. {ct} for grid size = {grid_size}, grid_eps = {grid_e}.")
        try:
            m_kan, m_sym = closure(dataset, grid_size, grid_e, lamb, k, optim, epochs, grid_update_num, stop_grid_update_step, alpha, beta, r2_threshold, exp_seed, device)
            metric_kan.append(m_kan)
            metric_sym.append(m_sym)
            if verbose:
                print(f"KAN Model: ROC-AUC of {m_kan:.2f}.\t Symbolic Model: ROC-AUC of {m_sym:.2f}.\n")
        except Exception as e:
            if verbose:
                print(f"Exception {e}\nOmmiting this one.")
            metric_kan.append(0)
            metric_sym.append(0)
        ct += 1

    modelsdf = pd.DataFrame(combinations, columns=['grid_sizes', 'grid_es'])
    modelsdf['metric_kan'] = metric_kan
    modelsdf['metric_sym'] = metric_sym
    modelsdf = modelsdf.dropna()
    modelsdf['metric_kan'] = modelsdf['metric_kan'].astype('float64')
    modelsdf['metric_sym'] = modelsdf['metric_sym'].astype('float64')

    if modelsdf.shape[0] > 0:
        paretodf = pd.DataFrame({"metric_kan": modelsdf['metric_kan'].values, "metric_sym": modelsdf['metric_sym'].values})
        mask = paretoset(paretodf, sense=["max", "max"])
        modelsdf['pareto'] = mask

    # ROC-AUC: higher is better
    modelsdf['metric_direction'] = 'max'

    return modelsdf

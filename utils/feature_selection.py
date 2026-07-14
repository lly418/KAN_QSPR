import torch
import itertools
import numpy as np
import pandas as pd
from kan import KAN
from paretoset import paretoset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, r2_score

from .data import get_data, get_data_return, get_multiply_label_data, get_multiply_label_data_new, get_data_with_mask
from .meter import Meter

def multiply_label_feature_selection(df, grid_size, grid_eps, k, thresholds, lambdas, optim="Adam", epochs=80, use_scaler=True,
                      data_split=(80, 10, 10), device='cuda', exp_seed=42, verbose=True,label_id=0):
    # Initialize a scaler if scaler=True
    if use_scaler == True:
        scaler = StandardScaler()
    else:
        scaler = None

    # Get the full dataset
    dataset = get_multiply_label_data(df, scaler=scaler, data_split=data_split, final_eval=False, feat_idxs=None, device=device,
                       exp_seed=exp_seed,label_id=label_id)

    # Get the combination of (threshold, lambda) pairs
    combinations = list(itertools.product(thresholds, lambdas))

    # Initialize lists
    features = []
    metric_values = []

    def closure(df, grid_size, grid_eps, k, dataset, threshold, lamb, optim, epochs, use_scaler, data_split, exp_seed,
                device):
        input_dim = dataset['train_input'].shape[1]
        output_dim = dataset['train_label'].unique().shape[0]

        # Train vanilla model
        model = KAN(width=[input_dim, output_dim], grid=grid_size, k=k, grid_eps=grid_eps, seed=exp_seed,
                    auto_save=False, device=device)

        results = model.fit(dataset, opt=optim, steps=epochs, update_grid=False, reg_metric='node_backward', lamb=lamb,
                            loss_fn=torch.nn.CrossEntropyLoss())
        # Prune inputs thanks to regularization
        model = model.prune_input(threshold=threshold, log_history=False)

        # Catalogue features that remain
        kept_feat_ids = (model.input_id).cpu().numpy()
        kept_feats = df.columns[kept_feat_ids].values

        if use_scaler == True:
            new_scaler = StandardScaler()
        else:
            new_scaler = None

        # Construct new dataset based on kept features
        new_data = get_multiply_label_data(df, scaler=new_scaler, data_split=data_split, feat_idxs=kept_feats, device=device,
                            exp_seed=exp_seed,label_id=label_id)

        new_input = new_data['train_input'].shape[1]
        new_output = new_data['train_label'].unique().shape[0]

        # Train new model, using only kept features
        new_model = KAN(width=[new_input, new_output], grid=grid_size, k=k, grid_eps=grid_eps, seed=exp_seed,
                        auto_save=False, device=device)
        new_results = new_model.fit(new_data, opt=optim, steps=epochs, update_grid=False, reg_metric='node_backward',
                                    lamb=0.0, loss_fn=torch.nn.CrossEntropyLoss())

        # Evaluate final model on validation data
        test_preds = torch.argmax(new_model.forward(new_data['val_input']).detach(), dim=1).cpu()
        truth = new_data['val_label'].cpu()

        # Calculate weighted f1-score
        metric = 100 * f1_score(truth, test_preds, average='weighted')

        del model
        del new_model

        return kept_feats, metric

    # Run loop for all combinations of lambda, threshold
    ct = 1
    for (threshold, lamb) in combinations:
        if verbose:
            print(f"Running Experiment No. {ct} for lambda = {lamb:.4f}, threshold = {threshold:.2f}.")
        try:
            feats, score = closure(df, grid_size, grid_eps, k, dataset, threshold, lamb, optim, epochs, use_scaler,
                                   data_split, exp_seed, device)

            features.append(feats)
            metric_values.append(score)

            if verbose:
                print(f"Kept {len(feats)} features and achieved Weighted F1-Score of {score:.2f}%.\n")
        except Exception as e:
            if verbose:
                print(f"Exception {e}\nOmmiting this one.")
            features.append([])
            metric_values.append(0)
        ct += 1

    # Gather results in dataframe, to be returned
    featsdf = pd.DataFrame(combinations, columns=['thresholds', 'lambdas'])
    featsdf['metric'] = np.array(metric_values)
    featsdf['features'] = features
    num_feats = featsdf['features'].apply(len)
    featsdf['num_feats'] = num_feats
    # Drop None values
    featsdf = featsdf.dropna()
    featsdf['metric'] = featsdf['metric'].astype('float64')

    if featsdf.shape[0] > 0:
        # Use results to find optimal lambda, threshold
        paretodf = pd.DataFrame({"num_feats": featsdf['num_feats'].values, "metric": featsdf['metric'].values})

        # Minimize number of features and maximize F1-Score
        mask = paretoset(paretodf, sense=["min", "max"])

        # Add a column to the DataFrame to distinguish Pareto set points
        featsdf['pareto'] = mask

    # Record metric direction: 'max' = higher is better, 'min' = lower is better
    featsdf['metric_direction'] = 'max'

    return featsdf

def feature_selection_roc_auc_return(df, grid_size, grid_eps, k, thresholds, lambdas, optim="Adam", epochs=80, use_scaler=True,
                      data_split=(80, 10, 10), device='cuda', exp_seed=42, verbose=True):
    """
    Performs feature selection for a given task.

    Args:
    -----
        df (pandas.core.frame.DataFrame):
            full data dataframe
        grid_size (int):
            size of grid for the KANs
        grid_eps (float):
            0.0 < grid_eps <= 1.0 - determines grid adaptability
        k (int):
            order of B-splines
        thresholds (array-like):
            array of all possible thresholds to be tested
        lambdas (array-like):
            array of all possible lambdas to be tested
        optim (string):
            either "LBFGS" or "Adam"
        epochs (int):
            number of steps for the optimizer during each training session
        use_scaler (bool):
            whether to scale the data or not
        data_split (tuple):
            tuple with percentages of train/val/test data - third value can be zero
        device (string):
            device on which the experiment will be run
        exp_seed (int):
            seed for reproducibility

    Returns:
    --------
        featsdf (pandas.core.frame.DataFrame):
            dataframe containing the full results of the grid search during feature selection

    """

    # Initialize a scaler if scaler=True
    if use_scaler == True:
        scaler = StandardScaler()
    else:
        scaler = None

    # Get the full dataset
    dataset = get_data(df, scaler=scaler, data_split=data_split, final_eval=False, feat_idxs=None, device=device,
                       exp_seed=exp_seed)
    # Convert labels to float32 for regression-style training
    dataset['train_label'] = dataset['train_label'].type(torch.float32).to(device).unsqueeze(1)
    dataset['val_label'] = dataset['val_label'].type(torch.float32).to(device). unsqueeze(1)
    dataset['test_label'] = dataset['test_label'].type(torch.float32).to(device).unsqueeze(1)
    # Get the combination of (threshold, lambda) pairs
    combinations = list(itertools.product(thresholds, lambdas))

    # Initialize lists
    features = []
    metric_values = []

    def closure(df, grid_size, grid_eps, k, dataset, threshold, lamb, optim, epochs, use_scaler, data_split, exp_seed,
                device):
        input_dim = dataset['train_input'].shape[1]
        output_dim = 1

        # Train vanilla model
        model = KAN(width=[input_dim, output_dim], grid=grid_size, k=k, grid_eps=grid_eps, seed=exp_seed,
                    auto_save=False, device=device)

        results = model.fit(dataset, opt=optim, steps=epochs, update_grid=False, reg_metric='node_backward', lamb=lamb,
                            loss_fn=torch.nn.MSELoss())
        # Prune inputs thanks to regularization
        model = model.prune_input(threshold=threshold, log_history=False)

        # Catalogue features that remain
        kept_feat_ids = (model.input_id).cpu().numpy()
        kept_feats = df.columns[kept_feat_ids].values

        if use_scaler == True:
            new_scaler = StandardScaler()
        else:
            new_scaler = None

        # Construct new dataset based on kept features
        new_data = get_data(df, scaler=new_scaler, data_split=data_split, feat_idxs=kept_feats, device=device,
                            exp_seed=exp_seed)
        new_data['train_label'] = new_data['train_label'].type(torch.float32).to(device).unsqueeze(1)
        new_data['val_label'] = new_data['val_label'].type(torch.float32).to(device).unsqueeze(1)
        new_data['test_label'] = new_data['test_label'].type(torch.float32).to(device).unsqueeze(1)

        new_input = new_data['train_input'].shape[1]
        new_output = 1

        # Train new model, using only kept features
        new_model = KAN(width=[new_input, new_output], grid=grid_size, k=k, grid_eps=grid_eps, seed=exp_seed,
                        auto_save=False, device=device)
        new_results = new_model.fit(new_data, opt=optim, steps=epochs, update_grid=False, reg_metric='node_backward',
                                    lamb=0.0, loss_fn=torch.nn.MSELoss())

        # Evaluate final model on validation data
        test_preds = new_model.forward(new_data['val_input']).detach().cpu()
        truth = new_data['val_label'].cpu()

        # Calculate ROC-AUC
        metric = roc_auc_score(truth, test_preds)

        del model
        del new_model

        return kept_feats, metric

    # Run loop for all combinations of lambda, threshold
    ct = 1
    for (threshold, lamb) in combinations:
        if verbose:
            print(f"Running Experiment No. {ct} for lambda = {lamb:.4f}, threshold = {threshold:.2f}.")
        try:
            feats, score = closure(df, grid_size, grid_eps, k, dataset, threshold, lamb, optim, epochs, use_scaler,
                                   data_split, exp_seed, device)

            features.append(feats)
            metric_values.append(score)

            if verbose:
                print(f"Kept {len(feats)} features and achieved ROC-AUC of {score:.2f}.\n")
        except Exception as e:
            if verbose:
                print(f"Exception {e}\nOmmiting this one.")
            features.append([])
            metric_values.append(0)
        ct += 1

    # Gather results in dataframe, to be returned
    featsdf = pd.DataFrame(combinations, columns=['thresholds', 'lambdas'])
    featsdf['metric'] = np.array(metric_values)
    featsdf['features'] = features
    num_feats = featsdf['features'].apply(len)
    featsdf['num_feats'] = num_feats
    # Drop None values
    featsdf = featsdf.dropna()
    featsdf['metric'] = featsdf['metric'].astype('float64')

    if featsdf.shape[0] > 0:
        # Use results to find optimal lambda, threshold
        paretodf = pd.DataFrame({"num_feats": featsdf['num_feats'].values, "metric": featsdf['metric'].values})

        # Minimize number of features and maximize ROC-AUC
        mask = paretoset(paretodf, sense=["min", "max"])

        # Add a column to the DataFrame to distinguish Pareto set points
        featsdf['pareto'] = mask

    # ROC-AUC: higher is better
    featsdf['metric_direction'] = 'max'

    return featsdf

def feature_selection_roc_auc_multiply_label(df, grid_size, grid_eps, k, thresholds, lambdas, optim="Adam", epochs=80, use_scaler=True,
                      data_split=(80, 10, 10), device='cuda', exp_seed=42, verbose=True):
    # Initialize a scaler if scaler=True
    if use_scaler == True:
        scaler = StandardScaler()
    else:
        scaler = None

    # Get the full dataset
    dataset = get_multiply_label_data_new(df, scaler=scaler, data_split=data_split, final_eval=False, feat_idxs=None, device=device,
                       exp_seed=exp_seed)
    # Convert labels to float32 for BCEWithLogitsLoss
    dataset['train_label'] = dataset['train_label'].type(torch.float32).to(device)
    dataset['val_label'] = dataset['val_label'].type(torch.float32).to(device)
    dataset['test_label'] = dataset['test_label'].type(torch.float32).to(device)
    # Get the combination of (threshold, lambda) pairs
    combinations = list(itertools.product(thresholds, lambdas))

    # Initialize lists
    features = []
    metric_values = []

    def closure(df, grid_size, grid_eps, k, dataset, threshold, lamb, optim, epochs, use_scaler, data_split, exp_seed,
                device):
        input_dim = dataset['train_input'].shape[1]
        output_dim = dataset['train_label'].shape[1]

        # Train vanilla model
        model = KAN(width=[input_dim, output_dim], grid=grid_size, k=k, grid_eps=grid_eps, seed=exp_seed,
                    auto_save=False, device=device)

        results = model.fit(dataset, opt=optim, steps=epochs, update_grid=False, reg_metric='node_backward', lamb=lamb,
                            loss_fn=torch.nn.BCEWithLogitsLoss())
        # Prune inputs thanks to regularization
        model = model.prune_input(threshold=threshold, log_history=False)

        # Catalogue features that remain
        kept_feat_ids = (model.input_id).cpu().numpy()
        kept_feats = df.columns[kept_feat_ids].values

        if use_scaler == True:
            new_scaler = StandardScaler()
        else:
            new_scaler = None

        # Construct new dataset based on kept features
        new_data = get_multiply_label_data_new(df, scaler=new_scaler, data_split=data_split, feat_idxs=kept_feats, device=device,
                            exp_seed=exp_seed)
        new_data['train_label'] = new_data['train_label'].type(torch.float32).to(device)
        new_data['val_label'] = new_data['val_label'].type(torch.float32).to(device)
        new_data['test_label'] = new_data['test_label'].type(torch.float32).to(device)

        new_input = new_data['train_input'].shape[1]
        new_output = new_data['train_label'].shape[1]

        # Train new model, using only kept features
        new_model = KAN(width=[new_input, new_output], grid=grid_size, k=k, grid_eps=grid_eps, seed=exp_seed,
                        auto_save=False, device=device)
        new_results = new_model.fit(new_data, opt=optim, steps=epochs, update_grid=False, reg_metric='node_backward',
                                    lamb=0.0, loss_fn=torch.nn.BCEWithLogitsLoss())

        # Evaluate final model on validation data
        test_preds = new_model.forward(new_data['val_input'])
        test_probs = torch.sigmoid(test_preds).cpu().detach().numpy()
        truth = new_data['val_label'].cpu()

        # Calculate ROC-AUC
        metric = roc_auc_score(truth, test_probs)

        del model
        del new_model

        return kept_feats, metric

    # Run loop for all combinations of lambda, threshold
    ct = 1
    for (threshold, lamb) in combinations:
        if verbose:
            print(f"Running Experiment No. {ct} for lambda = {lamb:.4f}, threshold = {threshold:.2f}.")
        try:
            feats, score = closure(df, grid_size, grid_eps, k, dataset, threshold, lamb, optim, epochs, use_scaler,
                                   data_split, exp_seed, device)

            features.append(feats)
            metric_values.append(score)

            if verbose:
                print(f"Kept {len(feats)} features and achieved ROC-AUC of {score:.2f}.\n")
        except Exception as e:
            if verbose:
                print(f"Exception {e}\nOmmiting this one.")
            features.append([])
            metric_values.append(0)
        ct += 1

    # Gather results in dataframe, to be returned
    featsdf = pd.DataFrame(combinations, columns=['thresholds', 'lambdas'])
    featsdf['metric'] = np.array(metric_values)
    featsdf['features'] = features
    num_feats = featsdf['features'].apply(len)
    featsdf['num_feats'] = num_feats
    # Drop None values
    featsdf = featsdf.dropna()
    featsdf['metric'] = featsdf['metric'].astype('float64')

    if featsdf.shape[0] > 0:
        # Use results to find optimal lambda, threshold
        paretodf = pd.DataFrame({"num_feats": featsdf['num_feats'].values, "metric": featsdf['metric'].values})

        # Minimize number of features and maximize ROC-AUC
        mask = paretoset(paretodf, sense=["min", "max"])

        # Add a column to the DataFrame to distinguish Pareto set points
        featsdf['pareto'] = mask

    # ROC-AUC: higher is better
    featsdf['metric_direction'] = 'max'

    return featsdf

def feature_selection_roc_auc(df, grid_size, grid_eps, k, thresholds, lambdas, optim="Adam", epochs=80, use_scaler=True,
                      data_split=(80, 10, 10), device='cuda', exp_seed=42, verbose=True):
    # Initialize a scaler if scaler=True
    if use_scaler == True:
        scaler = StandardScaler()
    else:
        scaler = None

    # Get the full dataset
    dataset = get_data(df, scaler=scaler, data_split=data_split, final_eval=False, feat_idxs=None, device=device,
                       exp_seed=exp_seed)

    # Get the combination of (threshold, lambda) pairs
    combinations = list(itertools.product(thresholds, lambdas))

    # Initialize lists
    features = []
    metric_values = []

    def closure(df, grid_size, grid_eps, k, dataset, threshold, lamb, optim, epochs, use_scaler, data_split, exp_seed,
                device):
        input_dim = dataset['train_input'].shape[1]
        output_dim = dataset['train_label'].unique().shape[0]

        # Train vanilla model
        model = KAN(width=[input_dim, output_dim], grid=grid_size, k=k, grid_eps=grid_eps, seed=exp_seed,
                    auto_save=False, device=device)

        results = model.fit(dataset, opt=optim, steps=epochs, update_grid=False, reg_metric='node_backward', lamb=lamb,
                            loss_fn=torch.nn.CrossEntropyLoss())
        # Prune inputs thanks to regularization
        model = model.prune_input(threshold=threshold, log_history=False)

        # Catalogue features that remain
        kept_feat_ids = (model.input_id).cpu().numpy()
        kept_feats = df.columns[kept_feat_ids].values

        if use_scaler == True:
            new_scaler = StandardScaler()
        else:
            new_scaler = None

        # Construct new dataset based on kept features
        new_data = get_data(df, scaler=new_scaler, data_split=data_split, feat_idxs=kept_feats, device=device,
                            exp_seed=exp_seed)

        new_input = new_data['train_input'].shape[1]
        new_output = new_data['train_label'].unique().shape[0]

        # Train new model, using only kept features
        new_model = KAN(width=[new_input, new_output], grid=grid_size, k=k, grid_eps=grid_eps, seed=exp_seed,
                        auto_save=False, device=device)
        new_results = new_model.fit(new_data, opt=optim, steps=epochs, update_grid=False, reg_metric='node_backward',
                                    lamb=0.0, loss_fn=torch.nn.CrossEntropyLoss())

        # Evaluate final model on validation data
        test_preds = new_model.forward(new_data['val_input'])
        pred_probs = torch.nn.functional.softmax(test_preds, dim=-1)
        pred_proba = pred_probs[:, 1].cpu().detach().numpy()
        truth = new_data['val_label'].cpu()

        # Calculate ROC-AUC
        metric = roc_auc_score(truth, pred_proba)

        del model
        del new_model

        return kept_feats, metric

    # Run loop for all combinations of lambda, threshold
    ct = 1
    for (threshold, lamb) in combinations:
        if verbose:
            print(f"Running Experiment No. {ct} for lambda = {lamb:.4f}, threshold = {threshold:.2f}.")
        try:
            feats, score = closure(df, grid_size, grid_eps, k, dataset, threshold, lamb, optim, epochs, use_scaler,
                                   data_split, exp_seed, device)

            features.append(feats)
            metric_values.append(score)

            if verbose:
                print(f"Kept {len(feats)} features and achieved ROC-AUC of {score:.2f}.\n")
        except Exception as e:
            if verbose:
                print(f"Exception {e}\nOmmiting this one.")
            features.append([])
            metric_values.append(0)
        ct += 1

    # Gather results in dataframe, to be returned
    featsdf = pd.DataFrame(combinations, columns=['thresholds', 'lambdas'])
    featsdf['metric'] = np.array(metric_values)
    featsdf['features'] = features
    num_feats = featsdf['features'].apply(len)
    featsdf['num_feats'] = num_feats
    # Drop None values
    featsdf = featsdf.dropna()
    featsdf['metric'] = featsdf['metric'].astype('float64')

    if featsdf.shape[0] > 0:
        # Use results to find optimal lambda, threshold
        paretodf = pd.DataFrame({"num_feats": featsdf['num_feats'].values, "metric": featsdf['metric'].values})

        # Minimize number of features and maximize ROC-AUC
        mask = paretoset(paretodf, sense=["min", "max"])

        # Add a column to the DataFrame to distinguish Pareto set points
        featsdf['pareto'] = mask

    # ROC-AUC: higher is better
    featsdf['metric_direction'] = 'max'

    return featsdf


def feature_selection(df, grid_size, grid_eps, k, thresholds, lambdas, optim="Adam", epochs=80, use_scaler=True, data_split=(80,10,10), device='cuda', exp_seed=42, verbose=True):
    """
    Performs feature selection for a given task.

    Args:
    -----
        df (pandas.core.frame.DataFrame):
            full data dataframe
        grid_size (int):
            size of grid for the KANs
        grid_eps (float):
            0.0 < grid_eps <= 1.0 - determines grid adaptability
        k (int):
            order of B-splines
        thresholds (array-like):
            array of all possible thresholds to be tested
        lambdas (array-like):
            array of all possible lambdas to be tested
        optim (string):
            either "LBFGS" or "Adam"
        epochs (int):
            number of steps for the optimizer during each training session
        use_scaler (bool):
            whether to scale the data or not
        data_split (tuple):
            tuple with percentages of train/val/test data - third value can be zero
        device (string):
            device on which the experiment will be run
        exp_seed (int):
            seed for reproducibility

    Returns:
    --------
        featsdf (pandas.core.frame.DataFrame):
            dataframe containing the full results of the grid search during feature selection

    """

    # Initialize a scaler if scaler=True
    if use_scaler == True:
        scaler = StandardScaler()
    else:
        scaler = None

    # Get the full dataset
    dataset = get_data(df, scaler=scaler, data_split=data_split, final_eval=False, feat_idxs=None, device=device, exp_seed=exp_seed)

    # Get the combination of (threshold, lambda) pairs
    combinations = list(itertools.product(thresholds, lambdas))

    # Initialize lists
    features = []
    metric_values = []

    def closure(df, grid_size, grid_eps, k, dataset, threshold, lamb, optim, epochs, use_scaler, data_split, exp_seed, device):
        input_dim = dataset['train_input'].shape[1]
        output_dim = dataset['train_label'].unique().shape[0]

        # Train vanilla model
        model = KAN(width=[input_dim, output_dim], grid=grid_size, k=k, grid_eps=grid_eps, seed=exp_seed, auto_save=False, device=device)

        results = model.fit(dataset, opt=optim, steps=epochs, update_grid=False, reg_metric='node_backward', lamb=lamb, loss_fn=torch.nn.CrossEntropyLoss())
        # Prune inputs thanks to regularization
        model = model.prune_input(threshold=threshold, log_history=False)

        # Catalogue features that remain
        kept_feat_ids = (model.input_id).cpu().numpy()
        kept_feats = df.columns[kept_feat_ids].values

        if use_scaler == True:
            new_scaler = StandardScaler()
        else:
            new_scaler = None

        # Construct new dataset based on kept features
        new_data = get_data(df, scaler=new_scaler, data_split=data_split, feat_idxs=kept_feats, device=device, exp_seed=exp_seed)

        new_input = new_data['train_input'].shape[1]
        new_output = new_data['train_label'].unique().shape[0]

        # Train new model, using only kept features
        new_model = KAN(width=[new_input, new_output], grid=grid_size, k=k, grid_eps=grid_eps, seed=exp_seed, auto_save=False, device=device)
        new_results = new_model.fit(new_data, opt=optim, steps=epochs, update_grid=False, reg_metric='node_backward', lamb=0.0, loss_fn=torch.nn.CrossEntropyLoss())

        # Evaluate final model on validation data
        test_preds = torch.argmax(new_model.forward(new_data['val_input']).detach(), dim=1).cpu()
        truth = new_data['val_label'].cpu()

        # Calculate weighted f1-score
        metric = 100*f1_score(truth, test_preds, average='weighted')

        del model
        del new_model

        return kept_feats, metric

    # Run loop for all combinations of lambda, threshold
    ct = 1
    for (threshold, lamb) in combinations:
        if verbose:
            print(f"Running Experiment No. {ct} for lambda = {lamb:.4f}, threshold = {threshold:.2f}.")
        try:
            feats, score = closure(df, grid_size, grid_eps, k, dataset, threshold, lamb, optim, epochs, use_scaler, data_split, exp_seed, device)

            features.append(feats)
            metric_values.append(score)

            if verbose:
                print(f"Kept {len(feats)} features and achieved Weighted F1-Score of {score:.2f}%.\n")
        except Exception as e:
            if verbose:
                print(f"Exception {e}\nOmmiting this one.")
            features.append([])
            metric_values.append(0)
        ct += 1

    # Gather results in dataframe, to be returned
    featsdf = pd.DataFrame(combinations, columns=['thresholds', 'lambdas'])
    featsdf['metric'] = np.array(metric_values)
    featsdf['features'] = features
    num_feats = featsdf['features'].apply(len)
    featsdf['num_feats'] = num_feats
    # Drop None values
    featsdf = featsdf.dropna()
    featsdf['metric'] = featsdf['metric'].astype('float64')

    if featsdf.shape[0] > 0:
        # Use results to find optimal lambda, threshold
        paretodf = pd.DataFrame({"num_feats": featsdf['num_feats'].values, "metric": featsdf['metric'].values})

        # Minimize number of features and maximize F1-Score
        mask = paretoset(paretodf, sense=["min", "max"])

        # Add a column to the DataFrame to distinguish Pareto set points
        featsdf['pareto'] = mask

    # Weighted F1: higher is better
    featsdf['metric_direction'] = 'max'

    return featsdf

def feature_selection_return(df, grid_size, grid_eps, k, thresholds, lambdas, optim="Adam", epochs=80, use_scaler=True, data_split=(80,10,10), device='cuda', exp_seed=42, verbose=True):
    """
    Performs feature selection for regression tasks (metric = RMSE, lower is better).
    """
    if use_scaler == True:
        scaler = StandardScaler()
    else:
        scaler = None

    # Get the full dataset
    dataset = get_data_return(df, scaler=scaler, data_split=data_split, final_eval=False, feat_idxs=None, device=device,
                       exp_seed=exp_seed)

    # Get the combination of (threshold, lambda) pairs
    combinations = list(itertools.product(thresholds, lambdas))

    # Initialize lists
    features = []
    metric_values = []

    def closure(df, grid_size, grid_eps, k, dataset, threshold, lamb, optim, epochs, use_scaler, data_split, exp_seed,
                device):
        input_dim = dataset['train_input'].shape[1]
        output_dim = 1

        loss_fn = torch.nn.MSELoss()

        # Train vanilla model
        model = KAN(width=[input_dim, output_dim], grid=grid_size, k=k, grid_eps=grid_eps, seed=exp_seed,
                    auto_save=False, device=device)

        results = model.fit(dataset, opt=optim, steps=epochs, update_grid=False, reg_metric='node_backward', lamb=lamb,
                            loss_fn=loss_fn)
        # Prune inputs thanks to regularization
        model = model.prune_input(threshold=threshold, log_history=False)

        # Catalogue features that remain
        kept_feat_ids = (model.input_id).cpu().numpy()
        kept_feats = df.columns[kept_feat_ids].values

        if use_scaler == True:
            new_scaler = StandardScaler()
        else:
            new_scaler = None

        # Construct new dataset based on kept features
        new_data = get_data_return(df, scaler=new_scaler, data_split=data_split, feat_idxs=kept_feats, device=device,
                            exp_seed=exp_seed)

        new_input = new_data['train_input'].shape[1]
        new_output = 1

        # Train new model, using only kept features
        new_model = KAN(width=[new_input, new_output], grid=grid_size, k=k, grid_eps=grid_eps, seed=exp_seed,
                        auto_save=False, device=device)
        new_results = new_model.fit(new_data, opt=optim, steps=epochs, update_grid=False, reg_metric='node_backward',
                                    lamb=0.0, loss_fn=loss_fn)

        # Evaluate on validation data — compute RMSE (lower is better)
        test_preds = new_model.forward(new_data['val_input']).detach().cpu().squeeze()
        truth = new_data['val_label'].cpu()

        metric = torch.sqrt(torch.mean((test_preds - truth) ** 2)).item()

        del model
        del new_model

        return kept_feats, metric

    # Run loop for all combinations of lambda, threshold
    ct = 1
    for (threshold, lamb) in combinations:
        if verbose:
            print(f"Running Experiment No. {ct} for lambda = {lamb:.4f}, threshold = {threshold:.2f}.")
        try:
            feats, score = closure(df, grid_size, grid_eps, k, dataset, threshold, lamb, optim, epochs, use_scaler,
                                   data_split, exp_seed, device)

            features.append(feats)
            metric_values.append(score)

            if verbose:
                print(f"Kept {len(feats)} features and achieved RMSE of {score:.4f}.\n")
        except Exception as e:
            if verbose:
                print(f"Exception {e}\nOmmiting this one.")
            features.append([])
            metric_values.append(0)
        ct += 1

    # Gather results in dataframe, to be returned
    featsdf = pd.DataFrame(combinations, columns=['thresholds', 'lambdas'])
    featsdf['metric'] = np.array(metric_values)
    featsdf['features'] = features
    num_feats = featsdf['features'].apply(len)
    featsdf['num_feats'] = num_feats
    # Drop None values
    featsdf = featsdf.dropna()
    featsdf['metric'] = featsdf['metric'].astype('float64')

    if featsdf.shape[0] > 0:
        # Use results to find optimal lambda, threshold
        paretodf = pd.DataFrame({"num_feats": featsdf['num_feats'].values, "metric": featsdf['metric'].values})

        # Minimize number of features AND minimize RMSE
        mask = paretoset(paretodf, sense=["min", "min"])

        # Add a column to the DataFrame to distinguish Pareto set points
        featsdf['pareto'] = mask

    # RMSE: lower is better
    featsdf['metric_direction'] = 'min'

    return featsdf

def feature_selection_return_R2(df, grid_size, grid_eps, k, thresholds, lambdas, optim="Adam", epochs=80, use_scaler=True, data_split=(80,10,10), device='cuda', exp_seed=42, verbose=True):
    """
    Performs feature selection for regression tasks (metric = R², higher is better).
    """
    if use_scaler == True:
        scaler = StandardScaler()
    else:
        scaler = None

    # Get the full dataset
    dataset = get_data_return(df, scaler=scaler, data_split=data_split, final_eval=False, feat_idxs=None, device=device,
                       exp_seed=exp_seed)

    # Get the combination of (threshold, lambda) pairs
    combinations = list(itertools.product(thresholds, lambdas))

    # Initialize lists
    features = []
    metric_values = []

    def closure(df, grid_size, grid_eps, k, dataset, threshold, lamb, optim, epochs, use_scaler, data_split, exp_seed,
                device):
        input_dim = dataset['train_input'].shape[1]
        output_dim = 1

        loss_fn = torch.nn.MSELoss()

        # Train vanilla model
        model = KAN(width=[input_dim, output_dim], grid=grid_size, k=k, grid_eps=grid_eps, seed=exp_seed,
                    auto_save=False, device=device)

        results = model.fit(dataset, opt=optim, steps=epochs, update_grid=False, reg_metric='node_backward', lamb=lamb,
                            loss_fn=loss_fn)
        # Prune inputs thanks to regularization
        model = model.prune_input(threshold=threshold, log_history=False)

        # Catalogue features that remain
        kept_feat_ids = (model.input_id).cpu().numpy()
        kept_feats = df.columns[kept_feat_ids].values

        if use_scaler == True:
            new_scaler = StandardScaler()
        else:
            new_scaler = None

        # Construct new dataset based on kept features
        new_data = get_data_return(df, scaler=new_scaler, data_split=data_split, feat_idxs=kept_feats, device=device,
                            exp_seed=exp_seed)

        new_input = new_data['train_input'].shape[1]
        new_output = 1

        # Train new model, using only kept features
        new_model = KAN(width=[new_input,new_output], grid=grid_size, k=k, grid_eps=grid_eps, seed=exp_seed,
                        auto_save=False, device=device)
        new_results = new_model.fit(new_data, opt=optim, steps=epochs, update_grid=False, reg_metric='node_backward',
                                    lamb=0.0, loss_fn=loss_fn)

        # Evaluate on validation data — compute R² (higher is better)
        test_preds = new_model.forward(new_data['val_input']).detach().cpu().squeeze()
        truth = new_data['val_label'].cpu()

        metric = r2_score(truth, test_preds)

        del model
        del new_model

        return kept_feats, metric

    # Run loop for all combinations of lambda, threshold
    ct = 1
    for (threshold, lamb) in combinations:
        if verbose:
            print(f"Running Experiment No. {ct} for lambda = {lamb:.4f}, threshold = {threshold:.2f}.")
        try:
            feats, score = closure(df, grid_size, grid_eps, k, dataset, threshold, lamb, optim, epochs, use_scaler,
                                   data_split, exp_seed, device)

            features.append(feats)
            metric_values.append(score)

            if verbose:
                print(f"Kept {len(feats)} features and achieved R² of {score:.4f}.\n")
        except Exception as e:
            if verbose:
                print(f"Exception {e}\nOmmiting this one.")
            features.append([])
            metric_values.append(0)
        ct += 1

    # Gather results in dataframe, to be returned
    featsdf = pd.DataFrame(combinations, columns=['thresholds', 'lambdas'])
    featsdf['metric'] = np.array(metric_values)
    featsdf['features'] = features
    num_feats = featsdf['features'].apply(len)
    featsdf['num_feats'] = num_feats
    # Drop None values
    featsdf = featsdf.dropna()
    featsdf['metric'] = featsdf['metric'].astype('float64')

    if featsdf.shape[0] > 0:
        # Use results to find optimal lambda, threshold
        paretodf = pd.DataFrame({"num_feats": featsdf['num_feats'].values, "metric": featsdf['metric'].values})

        # Minimize number of features and maximize R²
        mask = paretoset(paretodf, sense=["min", "max"])

        # Add a column to the DataFrame to distinguish Pareto set points
        featsdf['pareto'] = mask

    # R²: higher is better
    featsdf['metric_direction'] = 'max'

    return featsdf

def feature_selection_return_two_layers(df, grid_size, grid_eps, k, thresholds, lambdas, optim="Adam", epochs=80, use_scaler=True, data_split=(80,10,10), device='cuda', exp_seed=42, verbose=True):
    """
    Performs feature selection for regression tasks with two hidden layers (metric = RMSE, lower is better).
    """
    if use_scaler == True:
        scaler = StandardScaler()
    else:
        scaler = None

    # Get the full dataset
    dataset = get_data_return(df, scaler=scaler, data_split=data_split, final_eval=False, feat_idxs=None, device=device,
                       exp_seed=exp_seed)

    # Get the combination of (threshold, lambda) pairs
    combinations = list(itertools.product(thresholds, lambdas))

    # Initialize lists
    features = []
    metric_values = []

    def closure(df, grid_size, grid_eps, k, dataset, threshold, lamb, optim, epochs, use_scaler, data_split, exp_seed,
                device):
        input_dim = dataset['train_input'].shape[1]
        output_dim = 1

        loss_fn = torch.nn.MSELoss()

        # Train vanilla model
        model = KAN(width=[input_dim,2,output_dim], grid=grid_size, k=k, grid_eps=grid_eps, seed=exp_seed,
                    auto_save=False, device=device)

        results = model.fit(dataset, opt=optim, steps=epochs, update_grid=False, reg_metric='node_backward', lamb=lamb,
                            loss_fn=loss_fn)
        # Prune inputs thanks to regularization
        model = model.prune_input(threshold=threshold, log_history=False)

        # Catalogue features that remain
        kept_feat_ids = (model.input_id).cpu().numpy()
        kept_feats = df.columns[kept_feat_ids].values

        if use_scaler == True:
            new_scaler = StandardScaler()
        else:
            new_scaler = None

        # Construct new dataset based on kept features
        new_data = get_data_return(df, scaler=new_scaler, data_split=data_split, feat_idxs=kept_feats, device=device,
                            exp_seed=exp_seed)

        new_input = new_data['train_input'].shape[1]
        new_output = 1

        # Train new model, using only kept features
        new_model = KAN(width=[new_input,2, new_output], grid=grid_size, k=k, grid_eps=grid_eps, seed=exp_seed,
                        auto_save=False, device=device)
        new_results = new_model.fit(new_data, opt=optim, steps=epochs, update_grid=False, reg_metric='node_backward',
                                    lamb=0.0, loss_fn=loss_fn)

        # Evaluate on validation data — compute RMSE (lower is better)
        test_preds = new_model.forward(new_data['val_input']).detach().cpu().squeeze()
        truth = new_data['val_label'].cpu()

        metric = torch.sqrt(torch.mean((test_preds - truth) ** 2)).item()

        del model
        del new_model

        return kept_feats, metric

    # Run loop for all combinations of lambda, threshold
    ct = 1
    for (threshold, lamb) in combinations:
        if verbose:
            print(f"Running Experiment No. {ct} for lambda = {lamb:.4f}, threshold = {threshold:.2f}.")
        try:
            feats, score = closure(df, grid_size, grid_eps, k, dataset, threshold, lamb, optim, epochs, use_scaler,
                                   data_split, exp_seed, device)

            features.append(feats)
            metric_values.append(score)

            if verbose:
                print(f"Kept {len(feats)} features and achieved RMSE of {score:.4f}.\n")
        except Exception as e:
            if verbose:
                print(f"Exception {e}\nOmmiting this one.")
            features.append([])
            metric_values.append(0)
        ct += 1

    # Gather results in dataframe, to be returned
    featsdf = pd.DataFrame(combinations, columns=['thresholds', 'lambdas'])
    featsdf['metric'] = np.array(metric_values)
    featsdf['features'] = features
    num_feats = featsdf['features'].apply(len)
    featsdf['num_feats'] = num_feats
    # Drop None values
    featsdf = featsdf.dropna()
    featsdf['metric'] = featsdf['metric'].astype('float64')

    if featsdf.shape[0] > 0:
        # Use results to find optimal lambda, threshold
        paretodf = pd.DataFrame({"num_feats": featsdf['num_feats'].values, "metric": featsdf['metric'].values})

        # Minimize number of features AND minimize RMSE
        mask = paretoset(paretodf, sense=["min", "min"])

        # Add a column to the DataFrame to distinguish Pareto set points
        featsdf['pareto'] = mask

    # RMSE: lower is better
    featsdf['metric_direction'] = 'min'

    return featsdf


def feature_selection_with_mask(df, grid_size, grid_eps, k, thresholds, lambdas, optim="Adam", epochs=80, use_scaler=True, data_split=(80,10,10), device='cuda', exp_seed=42, verbose=True):
    """Feature selection for multi-task classification with masks (ROC-AUC metric).
    Uses get_data_with_mask for data loading with mask support.
    KAN architecture uses hidden layer [input, 128, output].
    """
    if use_scaler:
        scaler = StandardScaler()
    else:
        scaler = None

    dataset = get_data_with_mask(df, scaler=scaler, data_split=data_split, final_eval=False, feat_idxs=None, device=device, exp_seed=exp_seed)
    combinations = list(itertools.product(thresholds, lambdas))

    features = []
    metric_values = []

    def closure(df, grid_size, grid_eps, k, dataset, threshold, lamb, optim, epochs, use_scaler, data_split, exp_seed, device):
        input_dim = dataset['train_input'].shape[1]
        output_dim = dataset['train_label'].unique().shape[0]

        model = KAN(width=[input_dim, 128, output_dim], grid=grid_size, k=k, grid_eps=grid_eps, seed=exp_seed, auto_save=False, device=device)
        results = model.fit(dataset, opt=optim, steps=epochs, update_grid=False, reg_metric='node_backward', lamb=lamb, loss_fn=torch.nn.CrossEntropyLoss())
        model = model.prune_input(threshold=threshold, log_history=False)

        kept_feat_ids = (model.input_id).cpu().numpy()
        kept_feats = df.columns[kept_feat_ids].values

        if use_scaler:
            new_scaler = StandardScaler()
        else:
            new_scaler = None

        new_data = get_data_with_mask(df, scaler=new_scaler, data_split=data_split, feat_idxs=kept_feats, device=device, exp_seed=exp_seed)
        new_input = new_data['train_input'].shape[1]
        new_output = dataset['train_label'].unique().shape[0]

        new_model = KAN(width=[new_input, 128, new_output], grid=grid_size, k=k, grid_eps=grid_eps, seed=exp_seed, auto_save=False, device=device)
        new_results = new_model.fit(new_data, opt=optim, steps=epochs, update_grid=False, reg_metric='node_backward', lamb=0.0, loss_fn=torch.nn.CrossEntropyLoss())

        test_preds = new_model.forward(new_data['val_input']).detach()
        truth = new_data['val_label'].cpu()
        masks = new_data['mask_val'].cpu()
        eval_meter = Meter()
        eval_meter.update(test_preds, truth, masks)
        scores = np.mean(eval_meter.roc_auc_score())

        del model
        del new_model
        return kept_feats, scores

    ct = 1
    for (threshold, lamb) in combinations:
        if verbose:
            print(f"Running Experiment No. {ct} for lambda = {lamb:.4f}, threshold = {threshold:.2f}.")
        try:
            feats, score = closure(df, grid_size, grid_eps, k, dataset, threshold, lamb, optim, epochs, use_scaler, data_split, exp_seed, device)
            features.append(feats)
            metric_values.append(score)
            if verbose:
                print(f"Kept {len(feats)} features and achieved ROC-AUC of {score:.2f}.\n")
        except Exception as e:
            if verbose:
                print(f"Exception {e}\nOmmiting this one.")
            features.append([])
            metric_values.append(0)
        ct += 1

    featsdf = pd.DataFrame(combinations, columns=['thresholds', 'lambdas'])
    featsdf['metric'] = np.array(metric_values)
    featsdf['features'] = features
    num_feats = featsdf['features'].apply(len)
    featsdf['num_feats'] = num_feats
    featsdf = featsdf.dropna()
    featsdf['metric'] = featsdf['metric'].astype('float64')

    if featsdf.shape[0] > 0:
        paretodf = pd.DataFrame({"num_feats": featsdf['num_feats'].values, "metric": featsdf['metric'].values})
        mask = paretoset(paretodf, sense=["min", "max"])
        featsdf['pareto'] = mask

    # ROC-AUC: higher is better
    featsdf['metric_direction'] = 'max'

    return featsdf

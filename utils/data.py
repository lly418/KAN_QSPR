import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def get_data_with_mask(df, scaler=None, data_split=(80, 10, 10), final_eval=False, feat_idxs=None, device='cuda', exp_seed=42):
    """Classification data loader with mask support (for fault diagnosis / multi-task).

    Returns dataset dict with train/val/test inputs, labels, and masks.
    Labels are float32, masks are float32.
    """
    X = df.drop(columns=['label', 'masks'])
    y = df.loc[:, ['label', 'masks']]

    if feat_idxs is not None:
        X = X[feat_idxs]

    first_split = 1.0 - (data_split[0] / sum(data_split))
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=first_split, random_state=exp_seed)

    second_split = 1.0 - (data_split[1] / (data_split[1] + data_split[2]))
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=second_split, random_state=exp_seed)

    X_train, y_train = X_train.values, y_train.values
    X_val, y_val = X_val.values, y_val.values
    X_test, y_test = X_test.values, y_test.values

    if final_eval:
        X_train = np.concatenate((X_train, X_val), axis=0)
        y_train = np.concatenate((y_train, y_val), axis=0)

    if scaler:
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_val_scaled = X_val
        X_test_scaled = X_test

    mask_train = np.array([np.array(x, dtype=np.float32) for x in y_train[:, 1]])[:, np.newaxis]
    mask_val = np.array([np.array(x, dtype=np.float32) for x in y_val[:, 1]])[:, np.newaxis]
    mask_test = np.array([np.array(x, dtype=np.float32) for x in y_test[:, 1]])[:, np.newaxis]
    y_train = np.array([np.array(x, dtype=np.float32) for x in y_train[:, 0]])[:, np.newaxis]
    y_val = np.array([np.array(x, dtype=np.float32) for x in y_val[:, 0]])[:, np.newaxis]
    y_test = np.array([np.array(x, dtype=np.float32) for x in y_test[:, 0]])[:, np.newaxis]

    dataset = {}
    dataset['train_input'] = torch.from_numpy(X_train_scaled).type(torch.float32).to(device)
    dataset['train_label'] = torch.from_numpy(y_train).type(torch.float32).to(device)
    dataset['val_input'] = torch.from_numpy(X_val_scaled).type(torch.float32).to(device)
    dataset['val_label'] = torch.from_numpy(y_val).type(torch.float32).to(device)
    dataset['test_input'] = torch.from_numpy(X_test_scaled).type(torch.float32).to(device)
    dataset['test_label'] = torch.from_numpy(y_test).type(torch.float32).to(device)
    dataset['mask_train'] = torch.from_numpy(mask_train).type(torch.float32).to(device)
    dataset['mask_val'] = torch.from_numpy(mask_val).type(torch.float32).to(device)
    dataset['mask_test'] = torch.from_numpy(mask_test).type(torch.float32).to(device)

    return dataset


def get_data(df, scaler=None, data_split=(80, 10, 10), final_eval=False, feat_idxs=None, device='cuda', exp_seed=42):
    """Classification data loader (single-label, from utils_classify).

    Returns dataset dict with train/val/test inputs and labels.
    Labels are long tensors.
    """
    X = df.drop(columns=['label', 'masks'])
    y = df['label']

    if feat_idxs is not None:
        X = X[feat_idxs]

    first_split = 1.0 - (data_split[0] / sum(data_split))
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=first_split, random_state=exp_seed)

    second_split = 1.0 - (data_split[1] / (data_split[1] + data_split[2]))
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=second_split, random_state=exp_seed)

    X_train, y_train = X_train.values, y_train.values
    X_val, y_val = X_val.values, y_val.values
    X_test, y_test = X_test.values, y_test.values

    if final_eval:
        X_train = np.concatenate((X_train, X_val), axis=0)
        y_train = np.concatenate((y_train, y_val), axis=0)

    if scaler:
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_val_scaled = X_val
        X_test_scaled = X_test

    dataset = {}
    dataset['train_input'] = torch.from_numpy(X_train_scaled).type(torch.float32).to(device)
    dataset['train_label'] = torch.from_numpy(y_train).type(torch.long).to(device)
    dataset['val_input'] = torch.from_numpy(X_val_scaled).type(torch.float32).to(device)
    dataset['val_label'] = torch.from_numpy(y_val).type(torch.long).to(device)
    dataset['test_input'] = torch.from_numpy(X_test_scaled).type(torch.float32).to(device)
    dataset['test_label'] = torch.from_numpy(y_test).type(torch.long).to(device)

    return dataset


def get_data_return(df, scaler=None, data_split=(80, 10, 10), final_eval=False, feat_idxs=None, device='cuda', exp_seed=42):
    """Regression data loader.

    Returns dataset dict with train/val/test inputs and labels.
    Labels are float32 tensors shaped (N, 1).
    """
    X = df.drop(columns=['label', 'masks'])
    y = df['label']

    if feat_idxs is not None:
        X = X[feat_idxs]

    first_split = 1.0 - (data_split[0] / sum(data_split))
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=first_split, random_state=exp_seed)

    second_split = 1.0 - (data_split[1] / (data_split[1] + data_split[2]))
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=second_split, random_state=exp_seed)

    X_train, y_train = X_train.values, y_train.values
    X_val, y_val = X_val.values, y_val.values
    X_test, y_test = X_test.values, y_test.values

    if final_eval:
        X_train = np.concatenate((X_train, X_val), axis=0)
        y_train = np.concatenate((y_train, y_val), axis=0)

    if scaler:
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_val_scaled = X_val
        X_test_scaled = X_test

    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    dataset = {}
    dataset['train_input'] = torch.from_numpy(X_train_scaled).type(torch.float32).to(device)
    dataset['train_label'] = torch.from_numpy(y_train).type(torch.float32).to(device)
    dataset['val_input'] = torch.from_numpy(X_val_scaled).type(torch.float32).to(device)
    dataset['val_label'] = torch.from_numpy(y_val).type(torch.float32).to(device)
    dataset['test_input'] = torch.from_numpy(X_test_scaled).type(torch.float32).to(device)
    dataset['test_label'] = torch.from_numpy(y_test).type(torch.float32).to(device)

    return dataset


def get_multiply_label_data_new_with_mask(df, scaler=None, data_split=(80, 10, 10), final_eval=False, feat_idxs=None, device='cuda', exp_seed=42):
    """Multi-label classification data loader with mask support."""
    X = df.drop(columns=['label', 'masks'])
    y = df['label']

    if feat_idxs is not None:
        X = X[feat_idxs]

    first_split = 1.0 - (data_split[0] / sum(data_split))
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=first_split, random_state=exp_seed)

    second_split = 1.0 - (data_split[1] / (data_split[1] + data_split[2]))
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=second_split, random_state=exp_seed)

    X_train, y_train = X_train.values, y_train.values
    X_val, y_val = X_val.values, y_val.values
    X_test, y_test = X_test.values, y_test.values

    if final_eval:
        X_train = np.concatenate((X_train, X_val), axis=0)
        y_train = np.concatenate((y_train, y_val), axis=0)

    if scaler:
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_val_scaled = X_val
        X_test_scaled = X_test

    y_train = np.vstack([arr.astype(np.float32) for arr in y_train])
    y_val = np.vstack([arr.astype(np.float32) for arr in y_val])
    y_test = np.vstack([arr.astype(np.float32) for arr in y_test])

    if not final_eval:
        y_train = y_train[:, :]
        y_test = y_test[:, :]
        y_val = y_val[:, :]

    dataset = {}
    dataset['train_input'] = torch.from_numpy(X_train_scaled).type(torch.float32).to(device)
    dataset['train_label'] = torch.from_numpy(y_train).type(torch.long).to(device)
    dataset['val_input'] = torch.from_numpy(X_val_scaled).type(torch.float32).to(device)
    dataset['val_label'] = torch.from_numpy(y_val).type(torch.long).to(device)
    dataset['test_input'] = torch.from_numpy(X_test_scaled).type(torch.float32).to(device)
    dataset['test_label'] = torch.from_numpy(y_test).type(torch.long).to(device)

    return dataset


def get_multiply_label_data_new(df, scaler=None, data_split=(80, 10, 10), final_eval=False, feat_idxs=None, device='cuda', exp_seed=42):
    """Multi-label classification data loader."""
    X = df.drop(columns=['label', 'masks'])
    y = df['label']

    if feat_idxs is not None:
        X = X[feat_idxs]

    first_split = 1.0 - (data_split[0] / sum(data_split))
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=first_split, random_state=exp_seed)

    second_split = 1.0 - (data_split[1] / (data_split[1] + data_split[2]))
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=second_split, random_state=exp_seed)

    X_train, y_train = X_train.values, y_train.values
    X_val, y_val = X_val.values, y_val.values
    X_test, y_test = X_test.values, y_test.values

    if final_eval:
        X_train = np.concatenate((X_train, X_val), axis=0)
        y_train = np.concatenate((y_train, y_val), axis=0)

    if scaler:
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_val_scaled = X_val
        X_test_scaled = X_test

    y_train = np.vstack([arr.astype(np.float32) for arr in y_train])
    y_val = np.vstack([arr.astype(np.float32) for arr in y_val])
    y_test = np.vstack([arr.astype(np.float32) for arr in y_test])

    if not final_eval:
        y_train = y_train[:, :]
        y_test = y_test[:, :]
        y_val = y_val[:, :]

    dataset = {}
    dataset['train_input'] = torch.from_numpy(X_train_scaled).type(torch.float32).to(device)
    dataset['train_label'] = torch.from_numpy(y_train).type(torch.long).to(device)
    dataset['val_input'] = torch.from_numpy(X_val_scaled).type(torch.float32).to(device)
    dataset['val_label'] = torch.from_numpy(y_val).type(torch.long).to(device)
    dataset['test_input'] = torch.from_numpy(X_test_scaled).type(torch.float32).to(device)
    dataset['test_label'] = torch.from_numpy(y_test).type(torch.long).to(device)

    return dataset


def get_multiply_label_data(df, scaler=None, data_split=(80, 10, 10), final_eval=False, feat_idxs=None, device='cuda', exp_seed=42, label_id=0):
    """Multi-label classification data loader with label_id selection."""
    X = df.drop(columns=['label', 'masks'])
    y = df['label']

    if feat_idxs is not None:
        X = X[feat_idxs]

    first_split = 1.0 - (data_split[0] / sum(data_split))
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=first_split, random_state=exp_seed)

    second_split = 1.0 - (data_split[1] / (data_split[1] + data_split[2]))
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=second_split, random_state=exp_seed)

    X_train, y_train = X_train.values, y_train.values
    X_val, y_val = X_val.values, y_val.values
    X_test, y_test = X_test.values, y_test.values

    if final_eval:
        X_train = np.concatenate((X_train, X_val), axis=0)
        y_train = np.concatenate((y_train, y_val), axis=0)

    if scaler:
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_val_scaled = X_val
        X_test_scaled = X_test

    y_train = np.vstack([arr.astype(np.float32) for arr in y_train])
    y_val = np.vstack([arr.astype(np.float32) for arr in y_val])
    y_test = np.vstack([arr.astype(np.float32) for arr in y_test])

    if not final_eval:
        y_train = y_train[:, label_id]
        y_test = y_test[:, label_id]
        y_val = y_val[:, label_id]

    dataset = {}
    dataset['train_input'] = torch.from_numpy(X_train_scaled).type(torch.float32).to(device)
    dataset['train_label'] = torch.from_numpy(y_train).type(torch.long).to(device)
    dataset['val_input'] = torch.from_numpy(X_val_scaled).type(torch.float32).to(device)
    dataset['val_label'] = torch.from_numpy(y_val).type(torch.long).to(device)
    dataset['test_input'] = torch.from_numpy(X_test_scaled).type(torch.float32).to(device)
    dataset['test_label'] = torch.from_numpy(y_test).type(torch.long).to(device)

    return dataset

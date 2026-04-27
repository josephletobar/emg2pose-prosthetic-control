import numpy as np
import torch
from tqdm import tqdm
from emg2pose.feature_extraction import features, features_window
from emg2pose.data import Emg2PoseSessionData

def small_lstm_inference(data, model, seq_len, ds_factor, stride):

    X_raw = data['emg']
    y_raw = data['joint_angles']
    mask_raw = data.no_ik_failure

    model.eval()
    preds = []

    # --- downsample ---
    X = X_raw[::ds_factor]
    y = y_raw[::ds_factor]
    mask = mask_raw[::ds_factor]

    # --- build predictions ---
    total = (len(X) - seq_len) // stride
    for i in tqdm(range(0, len(X) - seq_len, stride), total=total, desc="LSTM inference"):
        window = X[i:i+seq_len]
        window = torch.tensor(window[None, ...], dtype=torch.float32)

        with torch.no_grad():
            pred = model(window).cpu().numpy()

        preds.append(pred[0])

    preds = np.array(preds)

    # --- align GT + mask EXACTLY like training ---
    y_gt = []
    mask_aligned = []

    total_gt = (len(y) - seq_len) // stride
    for i in range(0, len(y) - seq_len, stride):
        y_gt.append(y[i + seq_len - 1])
        mask_aligned.append(mask[i + seq_len - 1])

    y_gt = np.array(y_gt)
    mask_aligned = np.array(mask_aligned)

    return preds, y_gt, mask_aligned

def lstm_window_inference(window, model, ds_factor, gt_window, mask_window):
    """
    window: (T, C) raw EMG window
    gt_window: (T, D) raw joint angles aligned with window
    mask_window: (T,) raw mask aligned with window
    """

    # --- downsample everything consistently ---
    window_ds = window[::ds_factor]
    gt_ds = gt_window[::ds_factor]
    mask_ds = mask_window[::ds_factor]

    model.eval()

    x = torch.tensor(window_ds[None, ...], dtype=torch.float32)

    with torch.no_grad():
        pred = model(x).cpu().numpy()

    # --- match batch logic: last timestep of window ---
    gt = gt_ds[-1]
    mask = mask_ds[-1]

    return pred[0], gt, mask

def emg2pose_inferece(data: Emg2PoseSessionData, module):

    session = data

    start_idx = 0
    stop_idx = len(session) # eval on the whole session

    session_window = session[start_idx:stop_idx]

    # no_ik_failure is not a field so we slice separately
    no_ik_failure_window = session.no_ik_failure[start_idx:stop_idx]

    batch = {
        "emg": torch.Tensor([session_window["emg"].T]),
        "joint_angles": torch.Tensor([session_window["joint_angles"].T]),
        "no_ik_failure": torch.Tensor([no_ik_failure_window]),
    }

    batch = {k: v.to(module.device) for k, v in batch.items()}

    preds, joint_angles, no_ik_failure = module.forward(batch)

    # Algorithms that use the initial state for ground truth will do poorly
    # when the first joint angles are missing!
    if (joint_angles[:, 0] == 0).all():
        print(
            "Warning! Ground truth not available at first time step!"
        )

    # BCT --> TC (as numpy)
    preds = preds[0].T.detach().cpu().numpy()
    joint_angles = joint_angles[0].T.detach().cpu().numpy()

    # print("Predictions Shape: " + str(preds.shape))
    # print("Ground Truth Shape " + str(joint_angles.shape))

    return preds, joint_angles, no_ik_failure

def emg2pose_window_inference(window, module, gt_window, mask_window):
    """
    window: (T, C)
    gt_window: (T, D)
    mask_window: (T,)
    """

    # (T, C) -> (1, C, T)
    emg = torch.tensor(window.T, dtype=torch.float32).unsqueeze(0)

    batch = {
        "emg": emg,
        "joint_angles": torch.zeros((1, gt_window.shape[1], emg.shape[-1])),
        "no_ik_failure": torch.ones((1, emg.shape[-1]))
    }

    batch = {k: v.to(module.device) for k, v in batch.items()}

    with torch.no_grad():
        preds, _, _ = module.forward(batch)

    # (1, C_out, T) -> (T, C_out)
    preds = preds[0].T.detach().cpu().numpy()

    # --- align like LSTM: last timestep ---
    pred = preds[-1]
    gt = gt_window[-1]
    mask = mask_window[-1]

    return pred, gt, mask

def classic_ml_inference(data, ridge_model, svr_model, pls_model, lags=2):

    x_features, gt, mask = features(data)

    N = len(x_features)
    D = gt.shape[1]

    ridge_preds = []
    pls_preds = []

    # initialize with zeros (or first GT if you want teacher forcing)
    prev_ridge = [np.zeros(D) for _ in range(lags)]
    prev_pls   = [np.zeros(D) for _ in range(lags)]

    for t in range(N):

        if t < lags:
            ridge_preds.append(prev_ridge[-1])
            pls_preds.append(prev_pls[-1])
            continue

        ridge_input = np.concatenate([x_features[t], np.concatenate(prev_ridge)])
        pls_input   = np.concatenate([x_features[t], np.concatenate(prev_pls)])

        ridge_pred = ridge_model.predict(ridge_input[None, :])[0]
        pls_pred   = pls_model.predict(pls_input[None, :])[0]

        ridge_preds.append(ridge_pred)
        pls_preds.append(pls_pred)

        prev_ridge = [ridge_pred] + prev_ridge[:-1]
        prev_pls   = [pls_pred] + prev_pls[:-1]

    ridge_preds = np.array(ridge_preds)
    pls_preds   = np.array(pls_preds)

    svr_preds = None

    return ridge_preds, svr_preds, pls_preds, gt, mask

def ridge_window_inference(window, ridge_model, gt_window, mask_window):
    x = features_window(window)
    D = gt_window.shape[1]

    # init state on first call
    if not hasattr(ridge_model, "_prev"):
        ridge_model._prev = [np.zeros(D) for _ in range(2)]  # lags=2

    x_aug = np.concatenate([x, np.concatenate(ridge_model._prev)])
    pred = ridge_model.predict(x_aug[None, :])[0]

    ridge_model._prev = [pred] + ridge_model._prev[:-1]

    gt = gt_window[-1]
    mask = mask_window[-1]

    return pred, gt, mask


def svr_window_inference(window, svr_model, gt_window, mask_window):
    x_features = features_window(window)
    x_features = x_features[None, :]
    pred = svr_model.predict(x_features)[0]

    gt = gt_window[-1]
    mask = mask_window[-1]

    return pred, gt, mask


def pls_window_inference(window, pls_model, gt_window, mask_window):
    x = features_window(window)
    D = gt_window.shape[1]

    if not hasattr(pls_model, "_prev"):
        pls_model._prev = [np.zeros(D) for _ in range(2)]  # lags=2

    x_aug = np.concatenate([x, np.concatenate(pls_model._prev)])
    pred = pls_model.predict(x_aug[None, :])[0]

    pls_model._prev = [pred] + pls_model._prev[:-1]

    gt = gt_window[-1]
    mask = mask_window[-1]

    return pred, gt, mask
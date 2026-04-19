import numpy as np
import torch
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
    for i in range(0, len(X) - seq_len, stride):
        window = X[i:i+seq_len]
        window = torch.tensor(window[None, ...], dtype=torch.float32)

        with torch.no_grad():
            pred = model(window).cpu().numpy()

        preds.append(pred[0])

    preds = np.array(preds)

    # --- align GT + mask EXACTLY like training ---
    y_gt = []
    mask_aligned = []

    for i in range(0, len(y) - seq_len, stride):
        y_gt.append(y[i + seq_len - 1])
        mask_aligned.append(mask[i + seq_len - 1])

    y_gt = np.array(y_gt)
    mask_aligned = np.array(mask_aligned)

    return preds, y_gt, mask_aligned

def lstm_window_inference(window, model, ds_factor):
    """
    window: (T, C)
    """
    # downsample FIRST

    window = window[::ds_factor]

    model.eval()

    x = torch.tensor(window[None, ...], dtype=torch.float32)  # (1, T, C)

    with torch.no_grad():
        pred = model(x).cpu().numpy()  # (1, output_dim)

    return pred[0]  # (output_dim,)

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

def emg2pose_window_inference(window, module):
    """
    window: np.ndarray of shape (T, C)
    """

    # (T, C) -> (1, C, T)
    emg = torch.tensor(window.T, dtype=torch.float32).unsqueeze(0)

    batch = {
        "emg": emg,
        # dummy placeholders (required by forward)
        "joint_angles": torch.zeros((1, 20, emg.shape[-1])),  # adjust 20 if needed
        "no_ik_failure": torch.ones((1, emg.shape[-1]))
    }

    batch = {k: v.to(module.device) for k, v in batch.items()}

    with torch.no_grad():
        preds, _, _ = module.forward(batch)

    # (1, C_out, T) -> (T, C_out)
    preds = preds[0].T.detach().cpu().numpy()

    return preds

def classic_ml_inference(data, ridge_model, svr_model, pls_model):

    x_features, gt, mask = features(data)

    ridge_pred = ridge_model.predict(x_features)
    svr_pred = svr_model.predict(x_features)
    pls_pred = pls_model.predict(x_features)

    return ridge_pred, svr_pred, pls_pred, gt, mask

def ridge_window_inference(window, ridge_model):
    x_features = features_window(window)
    x_features = x_features[None, :]
    return ridge_model.predict(x_features)[0]

def svr_window_inference(window, svr_model):
    x_features = features_window(window)
    x_features = x_features[None, :]
    return svr_model.predict(x_features)[0]

def pls_window_inference(window, pls_model):
    x_features = features_window(window)
    x_features = x_features[None, :]
    return pls_model.predict(x_features)[0]
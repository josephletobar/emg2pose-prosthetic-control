from tqdm import tqdm
import torch
import numpy as np

def conv_lstm_inference(data, model, seq_len, ds_factor, stride):

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
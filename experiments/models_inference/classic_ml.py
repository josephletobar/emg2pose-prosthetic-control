import numpy as np
from emg2pose.feature_extraction import features, features_window

def classic_ml_inference(data, ridge_model, svr_model, pls_model, lags=2):

    x_features, gt, mask = features(data)

    N = len(x_features)
    D = gt.shape[1]

    ridge_preds = []
    pls_preds = []

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
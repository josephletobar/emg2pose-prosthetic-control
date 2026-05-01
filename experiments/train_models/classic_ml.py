from tqdm import tqdm
import numpy as np

from emg2pose.data import Emg2PoseSessionData
from emg2pose.feature_extraction import features

from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cross_decomposition import PLSRegression

from experiments.data_helpers import _concat_sessions

def build_features(user_train_dict):
    train_sessions_list = _concat_sessions(user_train_dict)

    X_all, y_all = [], []

    for session in tqdm(train_sessions_list, desc="Building features"):
        data = Emg2PoseSessionData(hdf5_path=session)
        X_feats, y_out, _ = features(data)

        X_all.append(X_feats)
        y_all.append(y_out)

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)

    return X, y

def _add_lags(X, y, lags=2):
    X_new = []
    y_new = []

    for t in range(lags, len(X)):
        lagged_y = y[t-lags:t][::-1].reshape(-1)
        X_new.append(np.concatenate([X[t], lagged_y]))
        y_new.append(y[t])

    return np.array(X_new), np.array(y_new)

def train_classic_ml(emg_features, joint_angles):

    X = emg_features
    y = joint_angles

    # --- ARX augmentation ---
    X, y = _add_lags(X, y, lags=2)

    # --- split (preserve time order) ---
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # --- RIDGE ---
    ridge_model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0))
    ])

    ridge_model.fit(X_train, y_train)

    # # --- SVR ---
    # svr_model = Pipeline([
    #     ("scaler", StandardScaler()),
    #     ("svr", MultiOutputRegressor(
    #         SVR(kernel="rbf", C=1.0, epsilon=0.1)
    #     ))
    # ])

    # svr_model.fit(X_train, y_train)
    svr_model = None

    # --- PLS  ---
    pls_model = PLSRegression(n_components=10)  # try 5–20 range if needed
    pls_model.fit(X_train, y_train)

    return ridge_model, svr_model, pls_model

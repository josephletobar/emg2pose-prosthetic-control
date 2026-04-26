from pathlib import Path
import os
import subprocess
import time
import shutil
import sys
import pandas as pd
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from emg2pose.data import Emg2PoseSessionData
from emg2pose.lightning import Emg2PoseModule
from emg2pose.utils import generate_hydra_config_from_overrides
from emg2pose.feature_extraction import features

from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cross_decomposition import PLSRegression

def _concat_sessions(user_train_dict):
    train_sessions_list = []

    # Loop over users and their respective sessions
    for user, sessions in user_train_dict.items():
        for session in sessions:
            train_sessions_list.append(session)

    return train_sessions_list

def _train_subset(sessions, data_download_dir, epochs=100, start_checkpoint=None):
    """
    Prepares a training dataset from a subset of sessions.

    Parameters
    ----------
    sessions : list
        List of session file paths to train on (e.g., .hdf5 files). Each entry should point
        to a single session on disk.

    data_download_dir : Path
        Directory where a temporary dataset folder will be created. This folder
        will contain copied session files, a generated metadata.csv, and any
        training artifacts such as checkpoints.

    Description
    -----------
    This function automates dataset construction by:
    1. Taking in a list of sessions to train on.
    2. Copying those sessions into a temporary directory.
    3. Generating a metadata.csv file describing train/val/test splits.
    4. Returning the prepared dataset directory for downstream training.

    The resulting directory can be passed directly to training pipelines.
    """

    TMP_DIR = data_download_dir / "emg_sessions"
    TMP_DIR.mkdir(exist_ok=True)

    n = len(sessions)
    if n == 0:
        raise ValueError("No sessions found")

    selected_sessions = sessions
    print(f"Training on {len(selected_sessions)} sessions")

    # clear temp folder
    for f in TMP_DIR.glob("*"):
        if f.is_file():
            f.unlink()
        elif f.is_dir():
            shutil.rmtree(f)

    rows = []

    for session_path in selected_sessions:
        filename = os.path.basename(session_path)
        print("Adding:", filename)

        shutil.copy(session_path, TMP_DIR / filename)

        filename_no_ext = filename.replace(".hdf5", "")
        session_name = filename_no_ext.split("-recording")[0]

        for split in ["train", "val", "test"]:
            rows.append({
                "session": session_name,
                "user": "debug_user",
                "stage": "debug",
                "start": 0,
                "end": 1,
                "side": "right",
                "filename": filename_no_ext,
                "moving_hand": "both",
                "held_out_user": False,
                "held_out_stage": False,
                "split": split,
                "generalization": "debug"
            })

    df = pd.DataFrame(rows)
    df.to_csv(TMP_DIR / "metadata.csv", index=False)

    print("Created metadata.csv at:", TMP_DIR)

    ckpt_root = data_download_dir / "checkpoints"
    ckpt_root.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = ckpt_root / f"run_{timestamp}"
    ckpt_dir.mkdir()

    ckpt_name = f"checkpoint_{timestamp}"

    cmd = [
        "python", "-m", "emg2pose.train",
        "train=True",
        "eval=True",
        "experiment=regression_vemg2pose",
        "+trainer.accelerator=cpu",
        "+trainer.devices=1",
        f"trainer.max_epochs={epochs}",
        f"+callbacks.1.dirpath={ckpt_dir}",
        f"+callbacks.1.filename={ckpt_name}",
        f"data_location={TMP_DIR}"
    ]

    if start_checkpoint is not None:
        cmd.append(f"checkpoint={start_checkpoint}")
        cmd.append("optimizer.lr=5e-5")

    subprocess.run(cmd, check=True)

    final_ckpt = ckpt_dir / f"{ckpt_name}.ckpt"
    return final_ckpt

# LSTM architecture and training; returns the trained model
def train_small_lstm(emg, joint_angles, epochs=1):

    X = emg
    y = joint_angles

    ds_factor = 4
    X = X[::ds_factor]
    y = y[::ds_factor]

    seq_len = 100
    target_samples = 20_000

    raw_N = len(X) - seq_len
    stride = max(1, raw_N // target_samples)

    def make_sequences(X, y, seq_len, stride):
        Xs, ys = [], []
        for i in range(0, len(X) - seq_len, stride):
            Xs.append(X[i:i+seq_len])
            ys.append(y[i+seq_len - 1])
        return np.array(Xs), np.array(ys)

    X_seq, y_seq = make_sequences(X, y, seq_len, stride)

    split_idx = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test  = torch.tensor(X_test, dtype=torch.float32)
    y_test  = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

    class ConvLSTM(nn.Module):
        def __init__(self, in_ch, out_dim):
            super().__init__()
            self.conv = nn.Conv1d(in_ch, 32, kernel_size=9, padding=4)
            self.relu = nn.ReLU()
            self.lstm = nn.LSTM(32, 128, num_layers=2, batch_first=True)
            self.fc = nn.Linear(128, out_dim)

        def forward(self, x):
            x = x.transpose(1, 2)
            x = self.relu(self.conv(x))
            x = x.transpose(1, 2)
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    model = ConvLSTM(X.shape[1], y.shape[1])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    last_loss = None

    for epoch in tqdm(range(epochs), desc="Training LSTM"):

        for xb, yb in train_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = loss

        tqdm.write(f"Epoch {epoch}: loss = {last_loss.item():.4f}")

    print(f"[LSTM] final_loss = {last_loss.item():.4f}")

    # IMPORTANT: return params for alignment
    return model, seq_len, ds_factor, stride


def get_emg2pose(data_dir):

    checkpoint_dir = Path(data_dir) / "emg2pose_model_checkpoints"

    # Download checkpoint if it does not exist
    if not checkpoint_dir.exists():
        os.system(f'''
        cd {data_dir} &&
        curl "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_model_checkpoints.tar.gz" -o emg2pose_model_checkpoints.tar.gz &&
        tar -xvzf emg2pose_model_checkpoints.tar.gz
        ''')

    config = generate_hydra_config_from_overrides(
        overrides=[
            "experiment=tracking_vemg2pose",
            f"checkpoint={data_dir}/emg2pose_model_checkpoints/regression_vemg2pose.ckpt"
        ]
    )

    module = Emg2PoseModule.load_from_checkpoint(
        config.checkpoint,
        network=config.network,
        optimizer=config.optimizer,
        lr_scheduler=config.lr_scheduler,
    )

    return module

def fine_tune_emg2pose(user_train_dict, data_dir, epochs=10):
    train_sessions_list = _concat_sessions(user_train_dict)

    meta_ckpt = Path(data_dir) / "emg2pose_model_checkpoints" / "regression_vemg2pose.ckpt"

    checkpoint = _train_subset(
        train_sessions_list,
        data_dir,
        epochs,
        start_checkpoint=meta_ckpt,
    )

    config = generate_hydra_config_from_overrides(
        overrides=[
            "experiment=regression_vemg2pose",
            f"checkpoint={checkpoint}",
        ]
    )

    return Emg2PoseModule.load_from_checkpoint(
        config.checkpoint,
        network=config.network,
        optimizer=config.optimizer,
        lr_scheduler=config.lr_scheduler,
    )

def train_emg2pose(user_train_dict, data_dir, epochs):

    train_sessions_list = _concat_sessions(user_train_dict)

    checkpoint = _train_subset(train_sessions_list, data_dir, epochs)

    config = generate_hydra_config_from_overrides(
        overrides=[
            "experiment=regression_vemg2pose",
            f"checkpoint={checkpoint}"
        ]
    )

    module = Emg2PoseModule.load_from_checkpoint(
        config.checkpoint,
        network=config.network,
        optimizer=config.optimizer,
        lr_scheduler=config.lr_scheduler,
    )
    
    return module

def build_features(user_train_dict):
    train_sessions_list = _concat_sessions(user_train_dict) # list of all training sessions

    X_all, y_all = [], [] # build feature dataset

    for session in train_sessions_list:
        data = Emg2PoseSessionData(hdf5_path=session)
        X_feats, y_out, _ = features(data)

        X_all.append(X_feats)
        y_all.append(y_out)

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)

    return X, y

def train_classic_ml(emg_features, joint_angles):

    X = emg_features
    y = joint_angles

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

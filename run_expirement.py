from pathlib import Path
import random
from pprint import pprint
import os

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from emg2pose.data import Emg2PoseSessionData
from emg2pose.train_subset import train_subset
from emg2pose.lightning import Emg2PoseModule
from emg2pose.utils import generate_hydra_config_from_overrides
from emg2pose.feature_extraction import features

from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cross_decomposition import PLSRegression

class ExperimentRunner():

    def __init__(self, data_regime, data_dir):

        # Data regime flag
        self.data_regime = data_regime

        # Dictionary containing user(s) and their respective session(s) to train models on
        self.user_train_dict = {}

        # File locations
        self.data_dir = data_dir
        self.metadata_df = pd.read_csv(data_dir / "emg2pose_metadata.csv")
        # User sessions
        self.user_list = sorted([
            p for p in Path(data_dir, "emg2pose_dataset_mini").iterdir()
            if p.is_dir()
        ])

    # User and Session Selection Helpers
    def _pick_one_user(self):
        rand_user = random.choice(self.user_list)
        return { rand_user: [] }
    def _random_subset(self, k):
        rand_users = random.sample(self.user_list, k)
        return {user: [] for user in rand_users}
    def _pick_sessions(self, user_train_dict):
        for user in user_train_dict:
            sessions = sorted(user.glob("*.hdf5"))
            if self.data_regime == "single_session":
                user_train_dict[user] = [random.choice(sessions)]
            else:
                user_train_dict[user] = sessions
        return user_train_dict

    # Load Data Based on Training Regime
    def _load_data(self):
        if self.data_regime == "single_session":
            user_train_dict = self._pick_one_user()
            self.user_train_dict = self._pick_sessions(user_train_dict)

        elif self.data_regime == "single_user":
            user_train_dict = self._pick_one_user()
            self.user_train_dict = self._pick_sessions(user_train_dict)

        elif self.data_regime == "multi_user":
            user_train_dict = self._random_subset(k=len(self.user_list) // 2)
            self.user_train_dict = self._pick_sessions(user_train_dict)

        elif self.data_regime == "full":
            user_train_dict = {user: [] for user in self.user_list} 
            self.user_train_dict = self._pick_sessions(user_train_dict)

        return user_train_dict
    
    # Helper to concatenate sessions across users
    def _concat_sessions(self, user_train_dict):

        train_sessions_list = []

        # Loop over users and their respective sessions
        for user, sessions in user_train_dict.items():
            for session in sessions:
                train_sessions_list.append(session)

        return train_sessions_list

    # LSTM architecture and training; returns the trained model
    def train_small_lstm(self):

        print("\n -- Training LSTM --")

        # Load session(s) (RAW EMG)
        X_all, y_all = [], []
        
        # Loop over users and their respective sessions
        for user, sessions in self.user_train_dict.items():
            for session in sessions:
                data = Emg2PoseSessionData(hdf5_path=session)
                X_all.append(data['emg'])
                y_all.append(data['joint_angles'])

        X = np.concatenate(X_all, axis=0)
        y = np.concatenate(y_all, axis=0)

        # downsampling 
        ds_factor = 4   # 2000 Hz to 500 Hz 
        X = X[::ds_factor]
        y = y[::ds_factor]

        # sequence params
        seq_len = 100   # ~200 ms at 500 Hz
        target_samples = 20_000

        # adaptive stride
        raw_N = len(X) - seq_len
        stride = max(1, raw_N // target_samples)

        # sequence building
        def make_sequences(X, y, seq_len, stride):
            Xs, ys = [], []
            for i in range(0, len(X) - seq_len, stride):
                Xs.append(X[i:i+seq_len])
                ys.append(y[i+seq_len - 1])
            return np.array(Xs), np.array(ys)

        X_seq, y_seq = make_sequences(X, y, seq_len, stride)

        # split
        split_idx = int(0.8 * len(X_seq))
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

        # tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_test  = torch.tensor(X_test, dtype=torch.float32)
        y_test  = torch.tensor(y_test, dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=64,
            shuffle=True
        )

        test_loader = DataLoader(
            TensorDataset(X_test, y_test),
            batch_size=64
        )

        # --- Conv + LSTM model ---
        class ConvLSTM(nn.Module):
            def __init__(self, in_ch, out_dim):
                super().__init__()

                # keep temporal resolution (no stride)
                self.conv = nn.Conv1d(
                    in_channels=in_ch,
                    out_channels=32,
                    kernel_size=9,
                    stride=1,
                    padding=4
                )

                self.relu = nn.ReLU()

                self.lstm = nn.LSTM(
                    input_size=32,
                    hidden_size=128,
                    num_layers=2,
                    batch_first=True
                )

                self.fc = nn.Linear(128, out_dim)

            def forward(self, x):
                x = x.transpose(1, 2)      # (B, C, T)
                x = self.conv(x)           # (B, 32, T)
                x = self.relu(x)
                x = x.transpose(1, 2)      # (B, T, 32)

                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        model = ConvLSTM(X.shape[1], y.shape[1])

        # train
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        for epoch in range(5):
            for xb, yb in train_loader:
                pred = model(xb)
                loss = loss_fn(pred, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch}: loss = {loss.item():.4f}")

        # eval
        model.eval()
        preds = []

        with torch.no_grad():
            for xb, _ in test_loader:
                preds.append(model(xb))

        y_pred = torch.cat(preds).numpy()
        y_test = y_test.numpy()

        print("-- Finished Training LSTM -- \n")

        return model
    
    def small_lstm_inference(self, data, model, seq_len=100, ds_factor=4, stride=1):
        # --- inference on RAW EMG (MATCHES training pipeline) ---

        X_raw = data['emg']
        y_raw = data['joint_angles']

        model.eval()
        preds = []

        # --- SAME preprocessing as training ---
        X = X_raw[::ds_factor]
        y = y_raw[::ds_factor]

        for i in range(0, len(X) - seq_len, stride):
            window = X[i:i+seq_len]
            window = torch.tensor(window[None, ...], dtype=torch.float32)

            with torch.no_grad():
                pred = model(window).cpu().numpy()

            preds.append(pred[0])

        preds = np.array(preds)

        # --- GT aligned EXACTLY like training ---
        y_gt = []
        for i in range(0, len(y) - seq_len, stride):
            y_gt.append(y[i + seq_len - 1])

        y_gt = np.array(y_gt)

        return preds, y_gt
    
    def get_emg2pose(self):

        print("\n -- Loading Meta's Pretrained Model -- ")

        checkpoint_dir = Path(self.data_dir) / "emg2pose_model_checkpoints"

        # Download checkpoint if it does not exist
        if not checkpoint_dir.exists():
            os.system(f'''
            cd {self.data_dir} &&
            curl "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_model_checkpoints.tar.gz" -o emg2pose_model_checkpoints.tar.gz &&
            tar -xvzf emg2pose_model_checkpoints.tar.gz
            ''')

        config = generate_hydra_config_from_overrides(
            overrides=[
                "experiment=tracking_vemg2pose",
                f"checkpoint={self.data_dir}/emg2pose_model_checkpoints/regression_vemg2pose.ckpt"
            ]
        )

        module = Emg2PoseModule.load_from_checkpoint(
            config.checkpoint,
            network=config.network,
            optimizer=config.optimizer,
            lr_scheduler=config.lr_scheduler,
        )

        print("-- Loaded Meta's Pretrained Model -- \n")

        return module
    
    def train_emg2pose(self):

        print("\n -- Training emg2pose -- ")

        train_sessions_list = self._concat_sessions(self.user_train_dict)

        checkpoint = train_subset(train_sessions_list, self.data_dir)

        config = generate_hydra_config_from_overrides(
            overrides=[
                "experiment=tracking_vemg2pose",
                f"checkpoint={checkpoint}"
            ]
        )

        module = Emg2PoseModule.load_from_checkpoint(
            config.checkpoint,
            network=config.network,
            optimizer=config.optimizer,
            lr_scheduler=config.lr_scheduler,
        )
        
        print("--Trained emg2pose -- \n ")

        return module
    
    def emg2pose_inferece(self, data, module):

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
    
    def train_classic_ml(self):

        print("\n -- Training Classical ML Methods -- ")

        train_sessions_list = self._concat_sessions(self.user_train_dict) # list of all training sessions

        X_all, y_all = [], [] # build feature dataset

        for session in train_sessions_list:
            data = Emg2PoseSessionData(hdf5_path=session)
            X_feats, y_out = features(data)

            X_all.append(X_feats)
            y_all.append(y_out)

        X = np.concatenate(X_all, axis=0)
        y = np.concatenate(y_all, axis=0)

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

        # --- SVR ---
        svr_model = Pipeline([
            ("scaler", StandardScaler()),
            ("svr", MultiOutputRegressor(
                SVR(kernel="rbf", C=1.0, epsilon=0.1)
            ))
        ])

        svr_model.fit(X_train, y_train)

        # --- PLS  ---
        pls_model = PLSRegression(n_components=10)  # try 5–20 range if needed
        pls_model.fit(X_train, y_train)

        print("-- Trained Classical ML Methods -- \n")

        return ridge_model, svr_model, pls_model
    
    def classic_ml_inference(self, data, ridge_model, svr_model, pls_model):

        x_features, _ = features(data)

        ridge_pred = ridge_model.predict(x_features)
        svr_pred = svr_model.predict(x_features)
        pls_pred = pls_model.predict(x_features)

        return ridge_pred, svr_pred, pls_pred

    def run(self):
        self.user_train_dict =  self._load_data()

        print("\n=== DATA REGIME:", self.data_regime, "===")
        print(f"Users selected: {len(self.user_train_dict)}")

        for user, sessions in self.user_train_dict.items():
            print(f"  {user.name}: {len(sessions)} session(s)")

        # Train models
        small_lstm_model = self.train_small_lstm()
        meta_emg2pose_model = self.get_emg2pose()
        my_emg2pose_model = self.train_emg2pose()
        ridge_model, svr_model, pls_model = self.train_classic_ml()

        # # Run Inference
        # data = None
        # preds, y_gt = self.small_lstm_inference(data, small_lstm_model)
        # preds, joint_angles, no_ik_failure = self.emg2pose_inferece(data, meta_emg2pose_model)
        # preds, joint_angles, no_ik_failure = self.emg2pose_inferece(data, my_emg2pose_model)
        # ridge_pred, svr_pred, pls_pred = self.classic_ml_inference(data, ridge_model, svr_model, pls_model)

if __name__ == "__main__":
    import argparse

    DEFAULT_DATA_DIR = Path("/Volumes") / "Crucial X9" # local machine

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_regime",
        type=str,
        choices=["single_session", "single_user", "multi_user", "full"],
        default="single_session"
    )
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)

    args = parser.parse_args()

    runner = ExperimentRunner(
        data_regime=args.data_regime,
        data_dir=args.data_dir, 
        )

    runner.run()
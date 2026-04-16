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
from emg2pose.metrics import get_default_metrics
from emg2pose.utils import downsample

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

        # Sessions to evalaute on
        self.seen_user_sesion = []
        self.unseen_user_sesion = []

        # File locations
        self.data_dir = data_dir
        self.metadata_df = pd.read_csv(data_dir / "emg2pose_metadata.csv")
        # User sessions
        self.user_list = sorted([
            p for p in Path(data_dir, "emg_dataset/by_user").iterdir()
            if p.is_dir()
        ])

    # Training User and Session Selection Helpers
    def _user_has_valid_session(self, user):
        for s in user.glob("*.hdf5"):
            try:
                _ = Emg2PoseSessionData(hdf5_path=s)
                return True
            except:
                continue
        return False

    def _pick_one_user(self):
        while True:
            rand_user = random.choice(self.user_list)
            if self._user_has_valid_session(rand_user):
                return {rand_user: []}

    def _random_subset(self, k):
        selected = {}

        while len(selected) < k:
            rand_user = random.choice(self.user_list)

            if rand_user in selected:
                continue

            if self._user_has_valid_session(rand_user):
                selected[rand_user] = []

        return selected

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
            held_out_user = random.choice(self.user_list) # pick one user to hold out
            train_users = [u for u in self.user_list if u != held_out_user] # all others go into training

            user_train_dict = {user: [] for user in train_users} 
            self.user_train_dict = self._pick_sessions(user_train_dict)

        return user_train_dict
    
    def _eval_seen_user(self):
         # pick one random user
        rand_user = random.choice(list(self.user_train_dict.keys()))

        # get trained sessions for that user
        sessions = self.user_train_dict[rand_user]

        # pick one random session
        rand_session = random.choice(sessions)

        self.seen_user_sesion = rand_session
        return rand_session

    def _eval_unseen_user(self):
        # users NOT used in training
        unseen_users = [u for u in self.user_list if u not in self.user_train_dict]

        if not unseen_users:
            raise ValueError("No unseen users available (all users were used in training)")

        # pick random unseen user
        rand_user = random.choice(unseen_users)

        # get sessions for that user
        sessions = sorted(rand_user.glob("*.hdf5"))

        # pick random session
        rand_session = random.choice(sessions)

        self.unseen_user_session = rand_session
        return rand_session
    
    # Helper to concatenate sessions across users
    def _concat_sessions(self, user_train_dict):

        train_sessions_list = []

        # Loop over users and their respective sessions
        for user, sessions in user_train_dict.items():
            for session in sessions:
                train_sessions_list.append(session)

        return train_sessions_list

    # LSTM architecture and training; returns the trained model
    def train_small_lstm(self, epochs=5):

        print("\n -- Training LSTM --")

        X_all, y_all = [], []

        for user, sessions in self.user_train_dict.items():
            for session in sessions:
                data = Emg2PoseSessionData(hdf5_path=session)
                X_all.append(data['emg'])
                y_all.append(data['joint_angles'])

        X = np.concatenate(X_all, axis=0)
        y = np.concatenate(y_all, axis=0)

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

        for epoch in range(epochs):
            for xb, yb in train_loader:
                pred = model(xb)
                loss = loss_fn(pred, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch}: loss = {loss.item():.4f}")

        print("-- Finished Training LSTM -- \n")

        # IMPORTANT: return params for alignment
        return model, seq_len, ds_factor, stride
    
    def small_lstm_inference(self, data, model, seq_len, ds_factor, stride):

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
    
    def train_emg2pose(self, epochs):

        print("\n -- Training emg2pose -- ")

        train_sessions_list = self._concat_sessions(self.user_train_dict)

        checkpoint = train_subset(train_sessions_list, self.data_dir, epochs)

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
    
    def _convert_metrics(self, results):
        out = {}

        for k, v in results.items():
            val = v.item() if hasattr(v, "item") else v

            if "mae" in k:
                out[k + "_deg"] = np.degrees(val)

            elif "vel" in k or "acc" in k or "jerk" in k:
                out[k + "_deg"] = np.degrees(val)

            elif "distance" in k:
                out[k + "_mm"] = val

            else:
                out[k] = val

        return out
        
    def _get_metrics(self, preds, joint_angles, no_ik_failure):
        def to_tensor(x, is_mask=False):
            import torch, numpy as np

            if isinstance(x, np.ndarray):
                t = torch.tensor(x)
            elif isinstance(x, torch.Tensor):
                t = x
            else:
                t = torch.tensor(x)

            return t.bool() if is_mask else t.float()

        # --- convert ---
        pred = to_tensor(preds)
        target = to_tensor(joint_angles)
        mask = to_tensor(no_ik_failure, is_mask=True)

        # --- fix shapes to (B, C, T) and (B, T) ---
        try:
            if pred.ndim == 2:   # (T, C)
                pred = pred.T.unsqueeze(0)
            if target.ndim == 2:
                target = target.T.unsqueeze(0)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
        except:
            pass

        # --- align lengths (graceful degradation) ---
        try:
            T_pred = pred.shape[-1]
            T_target = target.shape[-1]
            T_mask = mask.shape[-1]

            min_T = min(T_pred, T_target, T_mask)

            if not (T_pred == T_target == T_mask):
                print(f"[Warning] Length mismatch (pred={T_pred}, target={T_target}, mask={T_mask}) -> truncating to {min_T}")

            pred = pred[..., :min_T]
            target = target[..., :min_T]
            mask = mask[..., :min_T]

        except:
            pass

        # --- force CPU ---
        try:
            pred = pred.cpu()
            target = target.cpu()
            mask = mask.cpu()
        except:
            pass

        # --- metrics ---
        metrics = get_default_metrics()
        results = {}

        for m in metrics:
            try:
                results.update(m(pred, target, mask, stage="eval"))
            except:
                continue  # skip broken metric

        return self._convert_metrics(results)

    def run(self):
        self.user_train_dict = self._load_data()

        print("\n=== DATA REGIME:", self.data_regime, "===")
        print(f"Users selected: {len(self.user_train_dict)}")

        for user, sessions in self.user_train_dict.items():
            print(user)
            print(f"  {user.name}: {len(sessions)} session(s)")        

        # Train models
        small_lstm_model, seq_len, ds_factor, stride = self.train_small_lstm(epochs=1)
        meta_emg2pose_model = self.get_emg2pose()
        my_emg2pose_model = self.train_emg2pose(epochs=5)
        ridge_model, svr_model, pls_model = self.train_classic_ml()

        # -------- SEEN USER --------
        print("\n===== SEEN USER =====")

        seen_eval_session = self._eval_seen_user()
        data = Emg2PoseSessionData(hdf5_path=seen_eval_session)

        # --- Small LSTM ---
        preds, y_gt, mask_lstm = self.small_lstm_inference(
            data, small_lstm_model, seq_len, ds_factor, stride
        )

        print("\n[Small LSTM]")
        print(self._get_metrics(preds, y_gt, mask_lstm))

        # --- Meta model ---
        preds, joint_angles, no_ik_failure = self.emg2pose_inferece(data, meta_emg2pose_model)
        mask = data.no_ik_failure

        print("\n[Meta EMG2Pose]")
        print(self._get_metrics(preds, joint_angles, no_ik_failure))

        # --- Your EMG2Pose ---
        preds, joint_angles, no_ik_failure = self.emg2pose_inferece(data, my_emg2pose_model)

        print("\n[My EMG2Pose]")
        print(self._get_metrics(preds, joint_angles, no_ik_failure))

        # --- Classical ML ---
        ridge_pred, svr_pred, pls_pred = self.classic_ml_inference(
            data, ridge_model, svr_model, pls_model
        )

        mask_30hz = downsample(mask.astype(float), 2000, 30) > 0.5

        print("\n[Ridge]")
        print(self._get_metrics(ridge_pred, y_gt, mask_30hz))

        print("\n[SVR]")
        print(self._get_metrics(svr_pred, y_gt, mask_30hz))

        print("\n[PLS]")
        print(self._get_metrics(pls_pred, y_gt, mask_30hz))


        # -------- UNSEEN USER --------
        print("\n===== UNSEEN USER =====")

        unseen_eval_session = self._eval_unseen_user()
        data = Emg2PoseSessionData(hdf5_path=unseen_eval_session)

        # --- Small LSTM ---
        preds, y_gt, mask_lstm = self.small_lstm_inference(
            data, small_lstm_model, seq_len, ds_factor, stride
        )

        print("\n[Small LSTM]")
        print(self._get_metrics(preds, y_gt, mask_lstm))

        # --- Meta model ---
        preds, joint_angles, no_ik_failure = self.emg2pose_inferece(data, meta_emg2pose_model)
        mask = data.no_ik_failure

        print("\n[Meta EMG2Pose]")
        print(self._get_metrics(preds, joint_angles, no_ik_failure))

        # --- Your EMG2Pose ---
        preds, joint_angles, no_ik_failure = self.emg2pose_inferece(data, my_emg2pose_model)

        print("\n[My EMG2Pose]")
        print(self._get_metrics(preds, joint_angles, no_ik_failure))

        # --- Classical ML ---
        ridge_pred, svr_pred, pls_pred = self.classic_ml_inference(
            data, ridge_model, svr_model, pls_model
        )

        mask_30hz = downsample(mask.astype(float), 2000, 30) > 0.5

        print("\n[Ridge]")
        print(self._get_metrics(ridge_pred, y_gt, mask_30hz))

        print("\n[SVR]")
        print(self._get_metrics(svr_pred, y_gt, mask_30hz))

        print("\n[PLS]")
        print(self._get_metrics(pls_pred, y_gt, mask_30hz))

        

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
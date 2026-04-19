from pathlib import Path
import random

import pandas as pd
import numpy as np
from emg2pose.data import Emg2PoseSessionData

from experiments.trainers import get_emg2pose, train_small_lstm, train_emg2pose, train_classic_ml, build_features
from experiments.load_data import load_data, concat_data
from experiments.metrics import get_experiment_metrics
from experiments.inference import small_lstm_inference, classic_ml_inference, emg2pose_inferece

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

    def eval_seen_user(self):
        # pick one random user
        rand_user = random.choice(list(self.user_train_dict.keys()))

        # get trained sessions for that user
        sessions = self.user_train_dict[rand_user]

        # pick one random session
        rand_session = random.choice(sessions)

        self.seen_user_sesion = rand_session
        return rand_session

    def eval_unseen_user(self):
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
    
    def run(self):

        self.user_train_dict = load_data(self.data_regime, self.user_list)

        print("\n=== DATA REGIME:", self.data_regime, "===")
        print(f"Users selected: {len(self.user_train_dict)}")

        for user, sessions in self.user_train_dict.items():
            print(user)
            print(f"  {user.name}: {len(sessions)} session(s)")        

        # Train models
        emg, joint_angles_lstm = concat_data(self.user_train_dict)
        small_lstm_model, seq_len, ds_factor, stride = train_small_lstm(emg, joint_angles_lstm, epochs=1)

        meta_emg2pose_model = get_emg2pose(self.data_dir)
        my_emg2pose_model = train_emg2pose(self.user_train_dict, self.data_dir, epochs=5)

        emg_features, joint_angles_ml = build_features(self.user_train_dict)
        ridge_model, svr_model, pls_model = train_classic_ml(emg_features, joint_angles_ml)

        # Run eval
        for label, session_function in [
            ("SEEN USER", self.eval_seen_user),
            ("UNSEEN USER", self.eval_unseen_user),
            ]:
            print(f"\n===== {label} EVAL =====")

            eval_session = session_function()
            data = Emg2PoseSessionData(hdf5_path=eval_session)

            # Small LSTM
            preds, y_gt, mask_lstm = small_lstm_inference(
                data, small_lstm_model, seq_len, ds_factor, stride
            )
            print("\n[Small LSTM]")
            print(get_experiment_metrics(preds, y_gt, mask_lstm))

            # Meta EMG2Pose
            preds, joint_angles, no_ik_failure = emg2pose_inferece(data, meta_emg2pose_model)
            print("\n[Meta EMG2Pose]")
            print(get_experiment_metrics(preds, joint_angles, no_ik_failure))

            # My EMG2Pose 
            preds, joint_angles, no_ik_failure = emg2pose_inferece(data, my_emg2pose_model)
            print("\n[My EMG2Pose]")
            print(get_experiment_metrics(preds, joint_angles, no_ik_failure))

            # Classical ML
            ridge_pred, svr_pred, pls_pred, gt, classic_ml_mask = classic_ml_inference(
                data, ridge_model, svr_model, pls_model
            )
            print("\n[Ridge]")
            print(get_experiment_metrics(ridge_pred, gt, classic_ml_mask))
            print("\n[SVR]")
            print(get_experiment_metrics(svr_pred, gt, classic_ml_mask))
            print("\n[PLS]")
            print(get_experiment_metrics(pls_pred, gt, classic_ml_mask))        

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
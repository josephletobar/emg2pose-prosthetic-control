from pathlib import Path
import random
from contextlib import contextmanager
import time
from datetime import datetime
import mediapy as media
import json

import pandas as pd
import numpy as np
from emg2pose.data import Emg2PoseSessionData
from emg2pose.visualization import remove_alpha_channel, joint_angles_to_frames_parallel
from emg2pose.utils import downsample

from experiments.trainers import get_emg2pose, train_small_lstm, train_emg2pose, train_classic_ml, build_features
from experiments.load_data import load_data, concat_data
from experiments.metrics import ExperimentMetrics, save_metrics_table
from experiments.stream_emg import stream_inference, save_latency_table
from experiments.inference import small_lstm_inference, classic_ml_inference, emg2pose_inferece, pls_window_inference, svr_window_inference, lstm_window_inference, ridge_window_inference, emg2pose_window_inference, features_window

DEFAULT_DATA_DIR = Path("/Volumes") / "Crucial X9" # local machine

@contextmanager
def timer(name):
    print(f"[START] {name}")
    t0 = time.perf_counter()
    yield
    t1 = time.perf_counter()
    print(f"[END] {name} | {t1 - t0:.2f} sec\n")

class ExperimentRunner():

    def __init__(self, data_regime, data_dir):

        # Data regime flag
        self.data_regime = data_regime

        # Save dir
        self.save_dir = f"results/{self.data_regime}/{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        # Dictionary containing user(s) and their respective session(s) to train models on
        self.user_train_dict = {}

        # Sessions to evalaute on
        self.seen_user_session = []
        self.held_out_session = None
        self.unseen_user_session = []

        # File locations
        self.data_dir = data_dir
        self.metadata_df = pd.read_csv(data_dir / "emg2pose_metadata.csv")
        # User sessions
        self.user_list = sorted([
            p for p in Path(data_dir, "emg_dataset/by_user").iterdir()
            if p.is_dir()
        ])

        self.MODEL_CONFIGS = {
            "lstm": {
                "native_fs": 500,
                "window_fn": lstm_window_inference,
                "WINDOW": None,   # filled at runtime (seq_len)
                "STRIDE": None,   # filled at runtime (stride)
            },
            "ridge": {
                "native_fs": 30,
                "window_fn": ridge_window_inference,
                "WINDOW": 500,
                "STRIDE": 67,
            },
            "svr": {
                "native_fs": 30,
                "window_fn": svr_window_inference,
                "WINDOW": 500,
                "STRIDE": 67,
            },
            "pls": {
                "native_fs": 30,
                "window_fn": pls_window_inference,
                "WINDOW": 500,
                "STRIDE": 67,
            },
            "meta_emg2pose": {
                "native_fs": 2000,
                "window_fn": emg2pose_window_inference,
                "WINDOW": 12000,
                "STRIDE": 50,
            },
            "my_emg2pose": {
                "native_fs": 2000,
                "window_fn": emg2pose_window_inference,
                "WINDOW": 12000,
                "STRIDE": 50,
            }
        }

        self.eval_data = None

        self.metrics_rows = []
        self.latency_rows = []

    def eval_seen_user(self):

        # sanity check on training data
        if self.data_regime == "test":
            return (self.user_train_dict[next(iter(self.user_train_dict))])[0]

        # a held out session from those we trained on
        return self.held_out_session

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
    
    def model_metrics(self, label, 
                    model_type, model,
                    preds, gt, mask,
                    **kwargs):
        
        m = ExperimentMetrics(preds, gt, mask)
        self.metrics_rows.append({
            "Test Set": label,
            "Model": model_type,
            **m.main
        })
        latency = stream_inference(self.eval_data, self.MODEL_CONFIGS[model_type]["window_fn"], model, self.MODEL_CONFIGS[model_type]["WINDOW"], self.MODEL_CONFIGS[model_type]["STRIDE"], **kwargs)

        self.latency_rows.append({
            "Model": model_type,
            **latency
        })

        print(f"  [{model_type} | {label}]")
        print("  metrics:")
        for k, v in m.main.items():
            print(f"    {k:25s}: {v:.4f}")
        print("  latency:")
        for k, v in latency.items():
            print(f"    {k:25s}: {v:.4f}")

        m.save_outputs(f"{self.save_dir}/{label}_{model_type}", self.MODEL_CONFIGS[model_type]["native_fs"])
    
    def run(self):

        self.user_train_dict, self.held_out_session = load_data(self.data_regime, self.user_list)

        Path(f"{self.save_dir}/train_dict.json").write_text(
            json.dumps({
                "train": {str(k): [str(s) for s in v] for k, v in self.user_train_dict.items()},
                "held_out": str(self.held_out_session)
            }, indent=2)
        )

        print("\n=== DATA REGIME:", self.data_regime, "===")
        print(f"Users selected: {len(self.user_train_dict)}")

        for user, sessions in self.user_train_dict.items():
            print(f"  {user.name}: {len(sessions)} session(s)\n")        

        # Train models
        with timer("LSTM Training"):
            emg, joint_angles_lstm = concat_data(self.user_train_dict)
            small_lstm_model, seq_len, ds_factor, stride = train_small_lstm(emg, joint_angles_lstm, epochs=5)

        with timer("Get Meta emg2pose"):
            meta_emg2pose_model = get_emg2pose(self.data_dir)

        # with timer("emg2pose Training"):
        #     my_emg2pose_model = train_emg2pose(self.user_train_dict, self.data_dir, epochs=5)

        with timer("Classical ML Training"):
            emg_features, joint_angles_ml = build_features(self.user_train_dict)
            ridge_model, _, pls_model = train_classic_ml(emg_features, joint_angles_ml)

        # Get metrics
        for label, session_function in [
            ("seen_user", self.eval_seen_user),
            ("unseen_user", self.eval_unseen_user),
            ]:
            if self.data_regime == "test" and label == "unseen_user": continue
            print(f"\n===== {label} EVAL ===== \n")

            eval_session = session_function()
            self.eval_data = Emg2PoseSessionData(hdf5_path=eval_session)
            frames = joint_angles_to_frames_parallel(downsample(self.eval_data["joint_angles"], 2000, 30)[0:250])
            frames = remove_alpha_channel(frames)
            media.write_video(f"{self.save_dir}/ground_truth_{label}_eval.mp4", frames, fps=30)
            print()
        
            # Small LSTM
            with timer("LSTM"):
                lstm_preds, lstm_gt, mask_lstm = small_lstm_inference(
                    self.eval_data, small_lstm_model, seq_len, ds_factor, stride
                )
                self.MODEL_CONFIGS["lstm"]["WINDOW"] = seq_len
                self.MODEL_CONFIGS["lstm"]["STRIDE"] = stride

                self.model_metrics(label, "lstm", small_lstm_model, lstm_preds, lstm_gt, mask_lstm, ds_factor=ds_factor)

            # Meta EMG2Pose
            with timer("Meta emg2pose"):
                preds, joint_angles, no_ik_failure = emg2pose_inferece(self.eval_data, meta_emg2pose_model)
                self.model_metrics(label, "meta_emg2pose", meta_emg2pose_model, preds, joint_angles, no_ik_failure)

            # # My EMG2Pose 
            # with timer("My emg2pose"):
            #     preds, joint_angles, no_ik_failure = emg2pose_inferece(self.eval_data, my_emg2pose_model)
            #     self.model_metrics(label, "my_emg2pose", my_emg2pose_model, preds, joint_angles, no_ik_failure)

            # Classical ML
            with timer("Classical ML"):
                ridge_pred, _, pls_pred, gt, classic_ml_mask = classic_ml_inference(
                    self.eval_data, ridge_model, None, pls_model
                )

            with timer("Ridge"):
                self.model_metrics(label, "ridge", ridge_model, ridge_pred, gt, classic_ml_mask)
            # with timer("SVR"):
            #     self.model_metrics(label, "svr", svr_model, svr_pred, gt, classic_ml_mask)
            with timer("PLS"):
                self.model_metrics(label, "pls", pls_model, pls_pred, gt, classic_ml_mask) 

        save_metrics_table(self.metrics_rows, self.save_dir)  
        save_latency_table(self.latency_rows, self.save_dir)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_regime",
        type=str,
        choices=["single_session", "single_user", "multi_user", "full"],
        default="test"
    )
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)

    args = parser.parse_args()

    runner = ExperimentRunner(
        data_regime=args.data_regime,
        data_dir=args.data_dir, 
        )

    runner.run()
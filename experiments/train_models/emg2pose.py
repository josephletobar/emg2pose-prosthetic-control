from pathlib import Path
import os
import subprocess
import shutil
import pandas as pd
from datetime import datetime
from pathlib import Path

from emg2pose.lightning import Emg2PoseModule
from emg2pose.utils import generate_hydra_config_from_overrides

from experiments.data_helpers import _concat_sessions

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

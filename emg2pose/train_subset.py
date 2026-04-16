
import subprocess
import os
import shutil
import pandas as pd
from datetime import datetime
from pathlib import Path

def train_subset(sessions, data_download_dir, epochs=100):
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

    # --- run training on full subset ---
    subprocess.run([
        "python", "-m", "emg2pose.train",
        "train=True",
        "eval=True",
        "experiment=tracking_vemg2pose",
        f"trainer.max_epochs={epochs}",
        f"+callbacks.1.dirpath={ckpt_dir}",
        f"+callbacks.1.filename={ckpt_name}",
        f"data_location={TMP_DIR}"
    ])

    final_ckpt = ckpt_dir / f"{ckpt_name}.ckpt"
    return final_ckpt

# emg2pose: Evaluation for Prosthetic Control

This project builds on the [emg2pose work by Facebook Research](https://github.com/facebookresearch/emg2pose) and extends its analysis to evaluate how EMG-based pose regression performs in prosthetic control contexts. It focuses on practical constraints such as latency, user variability, data availability, and model capacity, with an emphasis on requirements like low-latency inference and personalization.

> ## Data (Quoted)
> A dataset of Surface electromyography (sEMG) recordings paired with ground-truth, motion-capture recordings of the hands. Data loading, baseline model training, and baseline model evaluation code are provided.

<p align="center">
  <img src="https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_overview.png" alt="EMG2Pose Overview" width="75%">
</p>

> The entire dataset has 25,253 HDF5 files, each consisting of time-aligned, 2kHz sEMG and joint angles for a single hand in a single stage. Each stage is ~1 minute. There are 193 participants, spanning 370 hours and 29 stages. `emg2pose.data.Emg2PoseSessionData` offers a programmatic read-only interface into the HDF5 session files.  
>  

> The `metadata.csv` file includes the following information for each HDF5 file:  
>  
> | Column             | Description |
> |--------------------|-------------|
> | `user`              | Anonymized user ID |
> | `session`           | Recording session (there are multiple stages per recording session) |
> | `stage`             | Name of stage |
> | `side`              | Hand side (`left` or `right`) |
> | `moving_hand`       | Whether the hand is prompted to move during the stage |
> | `held_out_user`     | Whether the user is held out from the training set |
> | `held_out_stage`    | Whether the stage is held out from the training set |
> | `split`             | `train`, `test`, or `val` |
> | `generalization`    | Type of generalization; across user (`user`), stage (`stage`), or across user and stage (`user_stage`) |

> — Source: [Original emg2pose repository (Facebook Research)](https://github.com/facebookresearch/emg2pose)

### Download the Full Dataset (431 GiB)

```shell
# Download the full (431 GiB) version of the dataset, extract to ~/emg2pose_dataset
cd ~ && curl https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset.tar -o emg2pose_dataset.tar

# Unpack the tar to ~/emg2pose_dataset
tar -xvf emg2pose_dataset.tar
```

### Downloading Pre-trained Checkpoints

Meta provides pre-trained checkpoints (as `.ckpt` files) for the following:

1. vemg2pose (tracking, regression settings)

To download and unpack these checkpoints, run the following.

```shell
# Download checkpoints
cd ~ && curl "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_model_checkpoints.tar.gz" -o emg2pose_model_checkpoints.tar.gz

# Unpack to ~/emg2pose_model_checkpoints
tar -xvzf emg2pose_model_checkpoints.tar.gz
```

## Code
Code requirements are specified in `environment.yml`, which includes dependencies for this project as well as the original emg2pose library that this analysis builds upon. The file is included in the submission zip.

- `run_experiment.py`  
  Core experiment pipeline, including data loading, model training, and evaluation.

- `run.py`  
  Entry point for automated execution of `run_experiment.py` a specified number of times. Allows configuration of data regimes and models via CLI arguments.

- `avg_metrics.py`  
  Aggregates metrics produced by `run_experiment.py`, optionally filtered by a specific run date, and computes final reported results.

- `Experiments` folder  
  Helper modules for data processing, model architecture definitions, training and inference routines, and utility functions.

- `Notebooks` folder  (used for additional analysis supporting the paper)
  - `fine_tune.ipynb`  
    Fine-tunes the emg2pose model by freezing earlier layers and adapting it to a selected held-out user.
  - `subset_training.ipynb`  
    Trains the emg2pose model from scratch on a very small subset of data.



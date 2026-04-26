import numpy as np
import torch
from emg2pose.metrics import get_default_metrics

class ExperimentMetrics:
    def __init__(self, preds, joint_angles, no_ik_failure):
        self.preds = preds
        self.joint_angles = joint_angles
        self.no_ik_failure = no_ik_failure

        self._results = None 

    def _to_tensor(self, x, is_mask=False):
        if isinstance(x, np.ndarray):
            t = torch.tensor(x)
        elif isinstance(x, torch.Tensor):
            t = x
        else:
            t = torch.tensor(x)

        return t.bool() if is_mask else t.float()

    def _compute(self):

        pred = self._to_tensor(self.preds)
        target = self._to_tensor(self.joint_angles)
        mask = self._to_tensor(self.no_ik_failure, is_mask=True)

        # print("pred shape:", pred.shape)
        # print("target shape:", target.shape)

        # shape fix
        if pred.ndim == 2:
            pred = pred.T.unsqueeze(0)
        if target.ndim == 2:
            target = target.T.unsqueeze(0)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)

        # align lengths
        T = min(pred.shape[-1], target.shape[-1], mask.shape[-1])
        pred = pred[..., :T]
        target = target[..., :T]
        mask = mask[..., :T]

        pred = pred.cpu()
        target = target.cpu()
        mask = mask.cpu()

        metrics = get_default_metrics()
        results = {}

        for m in metrics:
            try:
                results.update(m(pred, target, mask, stage="eval"))
            except:
                continue

        self._results = self._convert(results)

    def _convert(self, results):
        out = {}

        for k, v in results.items():
            val = v.item() if hasattr(v, "item") else v

            if "mae" in k or "vel" in k or "acc" in k or "jerk" in k:
                out[k + "_deg"] = np.degrees(val)

            elif "distance" in k:
                out[k + "_mm"] = val

            else:
                out[k] = val

        return out
    
    def _downsample(self, native_fs, target_fs=30):
        from emg2pose.utils import downsample

        if native_fs == target_fs:
            return self.preds, self.joint_angles, self.no_ik_failure

        pred = downsample(self.preds, native_fs=native_fs, target_fs=30)
        target = downsample(self.joint_angles, native_fs=native_fs, target_fs=30)

        factor = int(native_fs / target_fs)
        mask = self.no_ik_failure[::factor]

        return pred, target, mask
    
    # default representative joints thumb, index, little
    def plot(self, native_fs, joints=[2, 6, 18]):
        import matplotlib.pyplot as plt

        pred, target, mask = self._downsample(native_fs)

        pred = self._to_tensor(pred)
        target = self._to_tensor(target)
        mask = self._to_tensor(mask, is_mask=True)

        if pred.ndim == 2:
            pred = pred.T.unsqueeze(0)
        if target.ndim == 2:
            target = target.T.unsqueeze(0)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)

        T = min(pred.shape[-1], target.shape[-1], mask.shape[-1])

        pred = pred[..., :T][0].cpu().numpy()     # (C, T)
        target = target[..., :T][0].cpu().numpy()
        mask = mask[..., :T][0].cpu().numpy()     # (T,)

        # apply mask
        pred = pred[:, mask]
        target = target[:, mask]

        # plotting
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        names = ["Thumb", "Index", "Little"]

        for i, j in enumerate(joints):
            axes[i].plot(target[j], label="GT")
            axes[i].plot(pred[j], label="Pred")
            axes[i].set_title(names[i])
            axes[i].legend()

        plt.tight_layout()
        return fig


    def save_video(self, save_path, native_fs, fps=15, max_frames=250):
        import numpy as np
        import emg2pose.visualization as vis
        import mediapy
        from joblib import Parallel, delayed
        from tqdm import tqdm

        pred, gt, _ = self._downsample(native_fs, fps)

        gt = gt[:max_frames]
        pred = pred[:max_frames]

        def render_frame(gt_ja, pred_ja):
            # render separately
            frame_gt = vis.fig_to_array(
                vis.plot_hand_mesh(gt_ja, color="lightpink", opacity=1)
            )
            frame_pred = vis.fig_to_array(
                vis.plot_hand_mesh(pred_ja, color="gray", opacity=1)
            )

            # remove alpha
            frame_gt = frame_gt[..., :3]
            frame_pred = frame_pred[..., :3]

            # side-by-side (width concat)
            return np.concatenate([frame_gt, frame_pred], axis=1)

        frames = Parallel(n_jobs=8)(
            delayed(render_frame)(gt[i], pred[i])
            for i in tqdm(list(range(len(gt))), desc="Rendering")
        )

        frames = np.array(frames)

        mediapy.write_video(save_path, frames, fps=fps)

        return frames
        
    def save_outputs(self, save_dir, native_fs):
        import os
        os.makedirs(save_dir, exist_ok=True)

        # save plot
        fig = self.plot(native_fs)  # modify plot() to return fig instead of just showing
        fig.savefig(f"{save_dir}/_plot.png", bbox_inches="tight")

        # save video
        self.save_video(f"{save_dir}/_visual.mp4", native_fs)

    @property
    def all(self):
        if self._results is None:
            self._compute()
        return self._results

    @property
    def main(self):
        r = self.all
        return {
            "eval_mae_deg": r.get("eval_mae_deg"),
            "eval_landmark_distance_mm": r.get("eval_landmark_distance_mm"),
            "eval_jerk_deg": r.get("eval_jerk_deg"),
        }
    

def save_metrics_table(rows, save_dir, name="metrics"):
    import os
    import pandas as pd

    os.makedirs(save_dir, exist_ok=True)

    df = pd.DataFrame(rows)

    # rename to paper-style headers
    df = df.rename(columns={
        "eval_mae_deg": "Angular Error (°)",
        "eval_landmark_distance_mm": "Landmark Distance (mm)",
        "eval_jerk_deg": "Jerk (°)"
    })

    # enforce column order if present
    cols = [
        "Test Set",
        "Model",
        "Angular Error (°)",
        "Landmark Distance (mm)",
        "Jerk (°)"
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    # save
    csv_path = f"{save_dir}/{name}.csv"
    latex_path = f"{save_dir}/{name}.tex"

    df.to_csv(csv_path, index=False)
    df.to_latex(latex_path, index=False)

    return df

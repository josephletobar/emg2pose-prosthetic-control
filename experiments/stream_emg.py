import numpy as np
import time
from emg2pose.data import Emg2PoseSessionData

class EmgStreamer:
    def __init__(self, data: Emg2PoseSessionData):
        self.emg = data['emg']
        self.gt = data['joint_angles']
        self.mask = data.no_ik_failure
        self.t = 0
        self.buffer = []

    def step(self):
        emg = self.emg[self.t]
        gt = self.gt[self.t]
        mask = self.mask[self.t]
        self.t += 1

        return emg, gt, mask
    
def stream_inference(
    data, func, model, WINDOW, STRIDE=50, MAX_STEPS=500, WARMUP=10,
    use_ema=False, alpha=0.2, **kwargs
):
    from tqdm import tqdm

    stream = EmgStreamer(data)
    size = data['emg'].shape[0]

    buf = []
    gt_buf_stream = []
    mask_buf_stream = []
    
    latency_buf = []
    pred_buf = []
    gt_buf = []
    mask_buf = []
    count = 0

    ema_pred = None 

    max_iters = min(size, MAX_STEPS * STRIDE)
    for i in tqdm(range(max_iters), desc="Streaming"):

        if count > MAX_STEPS:
            break

        emg_t, gt_t, mask_t = stream.step()

        buf.append(emg_t)
        gt_buf_stream.append(gt_t)
        mask_buf_stream.append(mask_t)

        if len(buf) > WINDOW:
            buf.pop(0)
            gt_buf_stream.pop(0)
            mask_buf_stream.pop(0)

        if len(buf) == WINDOW and i % STRIDE == 0:
            count += 1

            window = np.stack(buf, axis=0)
            gt_window = np.stack(gt_buf_stream, axis=0)
            mask_window = np.array(mask_buf_stream)

            t0 = time.perf_counter()

            preds, gt, mask = func(
                window,
                model,
                gt_window=gt_window,
                mask_window=mask_window,
                **kwargs
            )

            # EMA block 
            if use_ema:
                if ema_pred is None:
                    ema_pred = preds
                else:
                    ema_pred = alpha * preds + (1 - alpha) * ema_pred
                pred_buf.append(ema_pred)
            else:
                pred_buf.append(preds)

            gt_buf.append(gt)
            mask_buf.append(mask)

            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000
            latency_buf.append(latency_ms)

    mean = np.mean(latency_buf[WARMUP:])
    median = np.median(latency_buf[WARMUP:])

    pred_buf = np.stack(pred_buf, axis=0)
    gt_buf = np.stack(gt_buf, axis=0)
    mask_buf = np.array(mask_buf)

    return (
        {
            "mean_latency_ms": mean,
            "median_latency_ms": median
        },
        pred_buf, gt_buf, mask_buf
    )
            

def save_latency_table(rows, save_dir, name="latency"):
    import os
    import pandas as pd

    os.makedirs(save_dir, exist_ok=True)

    df = pd.DataFrame(rows)

    # rename to paper-style headers
    df = df.rename(columns={
        "mean_latency_ms": "Mean Latency (ms)",
        "median_latency_ms": "Median Latency (ms)",
    })

    # enforce column order
    cols = [
        "Model",
        "Mean Latency (ms)",
        "Median Latency (ms)",
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    csv_path = f"{save_dir}/{name}.csv"
    latex_path = f"{save_dir}/{name}.tex"

    df.to_csv(csv_path, index=False)
    df.to_latex(latex_path, index=False)

    return df
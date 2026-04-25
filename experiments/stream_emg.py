import numpy as np
import time
from emg2pose.data import Emg2PoseSessionData

class EmgStreamer:
    def __init__(self, data: Emg2PoseSessionData):
        self.emg = data['emg']
        self.t = 0
        self.buffer = []

    def step(self):
        sample = self.emg[self.t]
        self.t += 1

        self.buffer.append(sample)
        return sample
    
def stream_inference(data, func, model, WINDOW, STRIDE=50, MAX_STEPS=500, WARMUP=10, **kwargs):
    from tqdm import tqdm

    stream = EmgStreamer(data)
    size = data['emg'].shape[0]

    buf = []
    latency_buf = []
    pred_buf = []
    count = 0

    max_iters = min(size, MAX_STEPS * STRIDE)
    for i in tqdm(range(max_iters), desc="Streaming"):

        if count > MAX_STEPS: break

        emg_t = stream.step()
        buf.append(emg_t)

        # keep buffer fixed size
        if len(buf) > WINDOW:
            buf.pop(0)

        # only run when buffer full AND at interval
        if len(buf) == WINDOW and i % STRIDE == 0:
            count +=1 

            window = np.stack(buf, axis=0)  # (T, C)

            t0 = time.perf_counter()

            preds = func(window, model, **kwargs)
            pred_buf.append(preds)

            t1 = time.perf_counter()

            latency_ms = (t1 - t0) * 1000

            latency_buf.append(latency_ms)

    mean = np.mean(latency_buf[WARMUP:])
    median = np.median(latency_buf[WARMUP:])

    return {
        "mean_latency_ms": mean,
        "median_latency_ms": median
    }
            

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
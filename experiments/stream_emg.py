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

    stream = EmgStreamer(data)
    size = data['emg'].shape[0]

    buf = []
    latency_buf = []
    pred_buf = []
    count = 0

    for i in range(size):

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

    return mean, median, pred_buf
            
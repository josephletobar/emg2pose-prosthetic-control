from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ConvLSTM architecture 
class ConvLSTM(nn.Module):
    # def __init__(self, in_ch, out_dim):
    #     super().__init__()
    #     self.conv = nn.Conv1d(in_ch, 32, kernel_size=9, padding=4)
    #     self.relu = nn.ReLU()
    #     self.lstm = nn.LSTM(32, 128, num_layers=2, batch_first=True)
    #     self.fc = nn.Linear(128, out_dim)

    def __init__(self, in_ch, out_dim, autoregressive=False):
        super().__init__()
        self.autoregressive = autoregressive

        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=5, padding=2),  # compress
            nn.ReLU(),
            )
        self.lstm = nn.LSTM(16, 64, num_layers=1, batch_first=True)

        self.fc = nn.Linear(64, out_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
        
# LSTM training; returns the trained model
def train_conv_lstm(emg, joint_angles, epochs=5):

    X = emg
    y = joint_angles

    ds_factor = 2
    X = X[::ds_factor]
    y = y[::ds_factor]

    seq_len = 100
    stride = 1

    def make_sequences(X, y, seq_len, stride):
        Xs, ys = [], []
        for i in range(0, len(X) - seq_len, stride):
            Xs.append(X[i:i+seq_len])
            ys.append(y[i+seq_len - 1])
        return np.array(Xs), np.array(ys)

    X_seq, y_seq = make_sequences(X, y, seq_len, stride)
    X_seq = X_seq[:20000]
    y_seq = y_seq[:20000]

    split_idx = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test  = torch.tensor(X_test, dtype=torch.float32)
    y_test  = torch.tensor(y_test, dtype=torch.float32)

    # model = ConvLSTM(X.shape[1], y.shape[1])

    autoregressive = True
    in_ch = X.shape[1] + (y.shape[1] if autoregressive else 0)
    model = ConvLSTM(in_ch, y.shape[1], autoregressive=autoregressive)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=64,
        shuffle=not model.autoregressive
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    last_loss = None

    for epoch in tqdm(range(epochs), desc="Training LSTM"):

        for xb, yb in train_loader:
            if model.autoregressive:
                prev = torch.zeros((xb.shape[0], xb.shape[1], y.shape[1]))
                xb_input = torch.cat([xb, prev], dim=2)
                pred = model(xb_input)
            else:
                pred = model(xb)

            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = loss

        tqdm.write(f"Epoch {epoch}: loss = {last_loss.item():.4f}")

    print(f"[LSTM] final_loss = {last_loss.item():.4f}")

    # IMPORTANT: return params for alignment
    return model, seq_len, ds_factor, stride



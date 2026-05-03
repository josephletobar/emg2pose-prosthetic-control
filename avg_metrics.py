from pathlib import Path
import pandas as pd
from datetime import datetime

ROOT = Path("results/single_user")

# Set what dates run to average over
RUN_DATE = 20260502 # or None for all

metrics_all = []
latency_all = []

# loop over run folders
for run_dir in ROOT.iterdir():
    # skip non-directories
    if not run_dir.is_dir():
        continue

    # optional date filter
    if RUN_DATE is not None and not run_dir.name.startswith(RUN_DATE):
        continue

    metrics_path = run_dir / "metrics.csv"
    latency_path = run_dir / "latency.csv"

    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        df["run"] = run_dir.name
        metrics_all.append(df)

    if latency_path.exists():
        df = pd.read_csv(latency_path)
        df["run"] = run_dir.name
        latency_all.append(df)

# concatenate
metrics_df = pd.concat(metrics_all, ignore_index=True)
latency_df = pd.concat(latency_all, ignore_index=True)

# AVERAGE METRICS
metrics_avg = (
    metrics_df
    .groupby(["Test Set", "Model"], as_index=False)
    .mean(numeric_only=True)
)

# AVERAGE LATENCY
latency_avg = (
    latency_df
    .groupby(["Model"], as_index=False)
    .mean(numeric_only=True)
)

# save
now = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
metrics_avg.to_csv(f"results/avg_metrics_{now}.csv", index=False)
latency_avg.to_csv(f"results/avg_latency_{now}.csv", index=False)

print(f"Saved avg_metrics_{now}.csv and avg_latency_{now}.csv")
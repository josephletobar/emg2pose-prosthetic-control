from pathlib import Path
import pandas as pd

ROOT = Path("results/single_user")

metrics_all = []
latency_all = []

# loop over run folders
for run_dir in ROOT.iterdir():
    if not run_dir.is_dir():
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

# ---- AVERAGE METRICS ----
metrics_avg = (
    metrics_df
    .groupby(["Test Set", "Model"], as_index=False)
    .mean(numeric_only=True)
)

# ---- AVERAGE LATENCY ----
latency_avg = (
    latency_df
    .groupby(["Model"], as_index=False)
    .mean(numeric_only=True)
)

# save
metrics_avg.to_csv("results/avg_metrics.csv", index=False)
latency_avg.to_csv("results/avg_latency.csv", index=False)

print("Saved avg_metrics.csv and avg_latency.csv")
import subprocess
import time

N_RUNS = 50

for i in range(N_RUNS):
    print(f"\n===== RUN {i+1}/{N_RUNS} =====\n")

    try:
        subprocess.run([
            "python", "-m", "run_experiment",
            "--data_regime", "single_user"
        ], check=True)

    except Exception as e:
        print(f"[CRASHED] run {i+1}: {e}")
        continue

    # small sleep to avoid weird file / GPU / IO issues
    time.sleep(2)
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, HTML

def evaluate_predictions(preds, gt, save_prefix=None):

    N_COLS = 2
    N_ROWS = int(np.ceil(preds.shape[1] / N_COLS))

    # --- Plot 1: Pred vs GT per joint ---
    fig1, axs = plt.subplots(N_ROWS, N_COLS, figsize=(4*N_COLS, 2*N_ROWS))
    axs = axs.flatten()

    for i, ax in enumerate(axs):
        if i >= preds.shape[1]:
            ax.axis('off')
            continue

        ax.set_title(f"Joint {i}")
        ax.plot(gt[:, i], label="gt")
        ax.plot(preds[:, i], label="pred")
        ax.legend()

    fig1.suptitle("Predicted vs Ground Truth Joint Angles")
    plt.tight_layout()
    fig1.subplots_adjust(top=0.95)

    if save_prefix:
        fig1.savefig(f"{save_prefix}_pred_vs_gt.png", dpi=300)

    plt.show()

    # --- Angular error (wrapped) ---
    error = np.arctan2(np.sin(preds - gt), np.cos(preds - gt))

    # --- Metrics ---
    mae_rad = np.mean(np.abs(error))
    mae_deg = np.degrees(mae_rad)

    # --- Per-joint MAE (bar chart, degrees) ---
    mae_per_joint = np.degrees(np.mean(np.abs(error), axis=0))

    fig2, ax = plt.subplots()
    ax.bar(range(len(mae_per_joint)), mae_per_joint)
    ax.set_title("MAE per Joint (deg)")
    ax.set_xlabel("Joint")
    ax.set_ylabel("MAE (deg)")

    plt.tight_layout()

    if save_prefix:
        fig2.savefig(f"{save_prefix}_mae_per_joint.png", dpi=300)

    plt.show()

    # --- Velocity smoothness ---
    # velocity
    vel_true = np.diff(gt, axis=0)
    vel_pred = np.diff(preds, axis=0)

    # jerk (key for jitter)
    jerk_true = np.diff(vel_true, axis=0)
    jerk_pred = np.diff(vel_pred, axis=0)

    # magnitudes
    vel_true_mag = np.mean(np.linalg.norm(vel_true, axis=1))
    vel_pred_mag = np.mean(np.linalg.norm(vel_pred, axis=1))

    jerk_true_mag = np.mean(np.linalg.norm(jerk_true, axis=1))
    jerk_pred_mag = np.mean(np.linalg.norm(jerk_pred, axis=1))

    # --- Bold output (important metrics) ---
    display(HTML(f"""
    <div style="font-size:20px; font-weight:bold;">
    MAE (deg): {mae_deg:.2f}<br>
    Velocity (gt / pred): {vel_true_mag:.4f} / {vel_pred_mag:.4f}<br>
    Jerk (gt / pred): {jerk_true_mag:.4f} / {jerk_pred_mag:.4f}
    </div>
    """))
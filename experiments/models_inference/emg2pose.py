
from emg2pose.data import Emg2PoseSessionData
import torch

def emg2pose_inferece(data: Emg2PoseSessionData, module):

    session = data

    start_idx = 0
    stop_idx = len(session) # eval on the whole session

    session_window = session[start_idx:stop_idx]

    # no_ik_failure is not a field so we slice separately
    no_ik_failure_window = session.no_ik_failure[start_idx:stop_idx]

    batch = {
        "emg": torch.Tensor([session_window["emg"].T]),
        "joint_angles": torch.Tensor([session_window["joint_angles"].T]),
        "no_ik_failure": torch.Tensor([no_ik_failure_window]),
    }

    batch = {k: v.to(module.device) for k, v in batch.items()}

    preds, joint_angles, no_ik_failure = module.forward(batch)

    # Algorithms that use the initial state for ground truth will do poorly
    # when the first joint angles are missing!
    if (joint_angles[:, 0] == 0).all():
        print(
            "Warning! Ground truth not available at first time step!"
        )

    # BCT --> TC (as numpy)
    preds = preds[0].T.detach().cpu().numpy()
    joint_angles = joint_angles[0].T.detach().cpu().numpy()

    # print("Predictions Shape: " + str(preds.shape))
    # print("Ground Truth Shape " + str(joint_angles.shape))

    return preds, joint_angles, no_ik_failure

def emg2pose_window_inference(window, module, gt_window, mask_window):
    """
    window: (T, C)
    gt_window: (T, D)
    mask_window: (T,)
    """

    # (T, C) -> (1, C, T)
    emg = torch.tensor(window.T, dtype=torch.float32).unsqueeze(0)

    batch = {
        "emg": emg,
        "joint_angles": torch.zeros((1, gt_window.shape[1], emg.shape[-1])),
        "no_ik_failure": torch.ones((1, emg.shape[-1]))
    }

    batch = {k: v.to(module.device) for k, v in batch.items()}

    with torch.no_grad():
        preds, _, _ = module.forward(batch)

    # (1, C_out, T) -> (T, C_out)
    preds = preds[0].T.detach().cpu().numpy()

    # --- align like LSTM: last timestep ---
    pred = preds[-1]
    gt = gt_window[-1]
    mask = mask_window[-1]

    return pred, gt, mask
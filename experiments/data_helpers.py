import random
import numpy as np
from emg2pose.data import Emg2PoseSessionData

# Load Data Based on Training Regime
def load_data(data_regime, user_list, select_user=None):
    if data_regime == "single_session" or data_regime == "test":

        if select_user is not None: user_train_dict = _get_selected_user(select_user, user_list)
        else: user_train_dict = pick_one_user(user_list)
        user_train_dict, held_out_session = pick_sessions(data_regime, user_train_dict)

    elif data_regime == "single_user":
        if select_user is not None: user_train_dict = _get_selected_user(select_user, user_list)
        else: user_train_dict = pick_one_user(user_list)
        user_train_dict, held_out_session = pick_sessions(data_regime, user_train_dict)

    elif data_regime == "multi_user":
        user_train_dict = random_subset(user_list, k=len(user_list) // 2)
        user_train_dict, held_out_session = pick_sessions(data_regime, user_train_dict)

    elif data_regime == "full":
        user_train_dict = {user: [] for user in user_list} 
        user_train_dict, held_out_session = pick_sessions(data_regime, user_train_dict)

    return user_train_dict, held_out_session

# overrides random single user selection
def _get_selected_user(select_user, user_list):
    # find the selected user
    target_user = next((u for u in user_list if u.name == select_user), None)
    if target_user is None:
        raise ValueError("user does not exist")
    return {target_user : []}

# Training User and Session Selection Helpers
def _user_has_valid_session(user):
    for s in user.glob("*.hdf5"):
        try:
            _ = Emg2PoseSessionData(hdf5_path=s)
            return True
        except:
            continue
    return False

def pick_one_user(user_list):
    while True:
        rand_user = random.choice(user_list)
        if _user_has_valid_session(rand_user):
            return {rand_user: []}

def random_subset(user_list, k):
    selected = {}

    while len(selected) < k:
        rand_user = random.choice(user_list)

        if rand_user in selected:
            continue

        if _user_has_valid_session(rand_user):
            selected[rand_user] = []

    return selected

def pick_sessions(data_regime, user_train_dict):
    held_out_session = None

    # pick random user to have a session held out for seen-user-eval
    rand_user = random.choice(list(user_train_dict.keys()))

    for user in user_train_dict:
        all_sessions = sorted(user.glob("*.hdf5"))

        valid_sessions = []
        for s in all_sessions:
            try:
                _ = Emg2PoseSessionData(hdf5_path=s)
                valid_sessions.append(s)
            except:
                continue

        if not valid_sessions:
            raise RuntimeError(f"No valid sessions for {user}")

        if data_regime == "single_session" or data_regime == "test":
            user_train_dict[user] = [random.choice(valid_sessions)]
        else:
            if user == rand_user:
                held_out_session = random.choice(valid_sessions)
                user_train_dict[user] = [s for s in valid_sessions if s != held_out_session]
            else:
                user_train_dict[user] = valid_sessions

    return user_train_dict, held_out_session

# Helper to concatenate sessions across users
def concat_data(user_train_dict):

    X_all = []
    y_all = []

    # Loop over users and their respective sessions
    for user, sessions in user_train_dict.items():
        for session in sessions:
            data = Emg2PoseSessionData(hdf5_path=session)

            X_all.append(data['emg'])
            y_all.append(data['joint_angles'])

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)

    return X, y

# Helper to concat sessions across users as a list
def _concat_sessions(user_train_dict):
    train_sessions_list = []

    # Loop over users and their respective sessions
    for user, sessions in user_train_dict.items():
        for session in sessions:
            train_sessions_list.append(session)

    return train_sessions_list
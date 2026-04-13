from pathlib import Path
import pandas as pd
import random
from emg2pose.data import Emg2PoseSessionData

class ExperimentRunner():

    def __init__(self, data_regime, data_dir):

        self.data_regime = data_regime

        self.user_dict

        self.data_dir = data_dir
        self.metadata_df = pd.read_csv(data_dir / "emg2pose_metadata.csv")

        self.user_list = sorted([
            p for p in Path(data_dir, "emg2pose_dataset_mini").iterdir()
            if p.is_dir()
        ])

        self.data = self._load_data()

    # Random User and Session Selection Helpers
    def _pick_one_user(self):
        rand_user = random.choice(self.user_list)
        return { rand_user: [] }
    def _random_subset(self, k):
        rand_users = random.sample(self.user_list, k)
        return {user: [] for user in rand_users}
    def _pick_sessions(self, user_dict):
        for user in user_dict:
            sessions = sorted(user.glob("*.hdf5"))
            if self.data_regime == "single_session":
                user_dict[user] = [random.choice(sessions)]
            else:
                user_dict[user] = sessions
        return user_dict

    # Load Data Based on Training Regime
    def _load_data(self):
        if self.data_regime == "single_session":
            user_dict = self._pick_one_user()
            self.user_dict = self._pick_sessions(user_dict)

        elif self.data_regime == "single_user":
            user_dict = self._pick_one_user()
            self.user_dict = self._pick_sessions(user_dict)

        elif self.data_regime == "multi_user":
            user_dict = self._random_subset(k=len(self._all_users) // 2)
            self.user_dict = self._pick_sessions(user_dict)

        elif self.data_regime == "full":
            user_dict = {user: [] for user in self.user_list} 
            self.user_dict = self._pick_sessions(user_dict)
        
    def run(self):
        self._load_data()

if __name__ == "__main__":
    import argparse

    DEFAULT_DATA_DIR = Path("/Volumes") / "Crucial X9" # local machine

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_regime",
        type=str,
        choices=["single_session", "single_user", "multi_user", "full"],
        default="single_session"
    )
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)

    args = parser.parse_args()

    runner = ExperimentRunner(
        data_regime=args.data_regime,
        data_dir=args.data_dir, 
        )

    runner.run()
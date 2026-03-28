import numpy as np
import pandas as pd
import joblib


class TelecomEnvironment:
    def __init__(self, data_path="data/network_traffic.csv"):
        self.df = pd.read_csv(data_path)

        # Load trained ML model
        self.model = joblib.load("models/congestion_model.pkl")

        self.current_step = 0

        # 🔥 Reduced for faster training
        self.max_steps = 500

    def reset(self):
        self.current_step = 0
        return self._get_state()

    def _get_state(self):
        row = self.df.iloc[self.current_step]

        features = np.array([
            row["active_users"],
            row["bandwidth_usage"],
            row["latency"],
            row["packet_loss"],
            row["signal_strength"]
        ])

        # 🔥 FIX: Convert to DataFrame to remove sklearn warning
        features_df = pd.DataFrame([features], columns=[
            "active_users",
            "bandwidth_usage",
            "latency",
            "packet_loss",
            "signal_strength"
        ])

        congestion_prob = self.model.predict_proba(features_df)[0][1]

        state = np.append(features, congestion_prob)

        return state

    def step(self, action):
        row = self.df.iloc[self.current_step].copy()

        original_congestion = row["congestion"]

        # 🔥 ACTION EFFECTS
        if action == 1:  # Increase bandwidth
            row["bandwidth_usage"] *= 0.85
        elif action == 2:  # Reduce load
            row["active_users"] *= 0.90
        elif action == 3:  # Boost signal
            row["signal_strength"] *= 1.05
        # action 0 = do nothing

        # Prepare features again after action
        features = np.array([
            row["active_users"],
            row["bandwidth_usage"],
            row["latency"],
            row["packet_loss"],
            row["signal_strength"]
        ])

        # 🔥 FIX: Use DataFrame again
        features_df = pd.DataFrame([features], columns=[
            "active_users",
            "bandwidth_usage",
            "latency",
            "packet_loss",
            "signal_strength"
        ])

        new_congestion_prob = self.model.predict_proba(features_df)[0][1]

        new_congestion = 1 if new_congestion_prob > 0.5 else 0

        # 🔥 REWARD LOGIC
        if original_congestion == 1 and new_congestion == 0:
            reward = 10   # Good: reduced congestion
        elif original_congestion == 0 and new_congestion == 1:
            reward = -10  # Bad: created congestion
        else:
            reward = -1   # Neutral / no improvement

        self.current_step += 1
        done = self.current_step >= self.max_steps

        next_state = self._get_state()

        return next_state, reward, done


# 🔥 TEST BLOCK (Optional but useful)
if __name__ == "__main__":
    env = TelecomEnvironment()

    state = env.reset()
    print("Initial State:", state)

    next_state, reward, done = env.step(action=1)
    print("Next State:", next_state)
    print("Reward:", reward)
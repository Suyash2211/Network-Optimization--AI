import numpy as np
import pandas as pd
import os 
import matplotlib.pyplot as plt


class TelecomNetworkSimulator:
    def __init__(self, num_towers=20, hours=168):
        """
        num_towers: number of telecom towers
        hours: total simulation time (default 168 = 7 days hourly)
        """
        self.num_towers = num_towers
        self.hours = hours

    def generate_peak_multiplier(self, hour):
        """
        Simulates daily peak hour pattern.
        Peak between 7 PM - 11 PM.
        """
        hour_of_day = hour % 24

        if 19 <= hour_of_day <= 23:
            return np.random.uniform(1.5, 2.0)  # Peak boost
        elif 8 <= hour_of_day <= 11:
            return np.random.uniform(1.2, 1.4)  # Morning moderate
        else:
            return np.random.uniform(0.8, 1.1)  # Normal traffic

    def simulate(self):
        data = []

        for tower_id in range(self.num_towers):
            base_users = np.random.randint(200, 500)

            for hour in range(self.hours):
                peak_multiplier = self.generate_peak_multiplier(hour)

                active_users = base_users * peak_multiplier
                bandwidth_usage = active_users * np.random.uniform(0.5, 0.8)
                latency = 20 + (bandwidth_usage / 100)
                packet_loss = np.clip(bandwidth_usage / 1000, 0, 5)
                signal_strength = np.random.uniform(70, 100)

                base_condition = (
                    bandwidth_usage > 380 or
                    latency > 23 or
                    packet_loss > 0.8
                )   

                # Add small randomness (real-world uncertainty)
                if base_condition:
                    congestion = np.random.choice([1, 0], p=[0.9, 0.1])
                else:
                    congestion = np.random.choice([0, 1], p=[0.95, 0.05])

                data.append([
                    tower_id,
                    hour,
                    active_users,
                    bandwidth_usage,
                    latency,
                    packet_loss,
                    signal_strength,
                    congestion
                ])

        columns = [
            "tower_id",
            "hour",
            "active_users",
            "bandwidth_usage",
            "latency",
            "packet_loss",
            "signal_strength",
            "congestion"
        ]

        return pd.DataFrame(data, columns=columns)


if __name__ == "__main__":
    simulator = TelecomNetworkSimulator()
    df = simulator.simulate()

    print(df.head())

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/network_traffic.csv", index=False)

    print("\nTotal Records:", len(df))
    print("Total Congestion Cases:", df["congestion"].sum())
    print("Congestion Rate: {:.2f}%".format(
        (df["congestion"].sum() / len(df)) * 100
    ))

    tower_0 = df[df["tower_id"] == 0]

    # Active Users Plot
    plt.figure(figsize=(10, 5))
    plt.plot(tower_0["hour"], tower_0["active_users"])
    plt.title("Active Users Over Time (Tower 0)")
    plt.xlabel("Hour")
    plt.ylabel("Active Users")
    plt.show()

    # Congestion Plot
    plt.figure(figsize=(10, 4))
    plt.plot(tower_0["hour"], tower_0["congestion"])
    plt.title("Congestion Over Time (Tower 0)")
    plt.xlabel("Hour")
    plt.ylabel("Congestion (0/1)")
    plt.show()
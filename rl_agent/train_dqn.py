import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import os

from environment import TelecomEnvironment
from dqn_model import DQN


# 🔥 Hyperparameters
GAMMA = 0.99
LR = 0.001
BATCH_SIZE = 32
EPSILON = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.98
EPISODES = 50


# 🔥 Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)


def train():
    print("🚀 Starting DQN training...")

    env = TelecomEnvironment()

    state_dim = 6
    action_dim = 4

    model = DQN(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    replay_buffer = ReplayBuffer()

    global EPSILON

    # 🔥 Store rewards for graph
    rewards_history = []

    for episode in range(EPISODES):
        state = env.reset()
        state = torch.FloatTensor(state)

        total_reward = 0
        done = False

        while not done:
            # 🔥 Progress logs
            if env.current_step % 100 == 0:
                print(f"Episode {episode+1} | Step {env.current_step}")

            # 🔥 Epsilon-greedy
            if random.random() < EPSILON:
                action = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad():
                    q_values = model(state)
                    action = torch.argmax(q_values).item()

            next_state, reward, done = env.step(action)
            next_state = torch.FloatTensor(next_state)

            # 🔥 Reward clipping (important)
            reward = max(min(reward, 10), -10)

            replay_buffer.add((state, action, reward, next_state, done))

            state = next_state
            total_reward += reward

            # 🔥 Training step
            if replay_buffer.size() > BATCH_SIZE:
                batch = replay_buffer.sample(BATCH_SIZE)

                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.stack(states)
                next_states = torch.stack(next_states)

                actions = torch.tensor(actions)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.float32)

                q_values = model(states)
                next_q_values = model(next_states)

                target = rewards + GAMMA * torch.max(next_q_values, dim=1)[0] * (1 - dones)

                current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze()

                loss = loss_fn(current_q, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 🔥 Store reward
        rewards_history.append(total_reward)

        # 🔥 Decay epsilon
        if EPSILON > EPSILON_MIN:
            EPSILON *= EPSILON_DECAY

        print(f"Episode {episode+1} | Reward: {total_reward} | Epsilon: {EPSILON:.3f}")

     # 🔥 SAVE MODEL HERE (AFTER TRAINING LOOP)
    import os
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/dqn_model.pth")
    print("✅ DQN model saved successfully!")

    # 🔥 Plot graph (VERY IMPORTANT)
    plt.figure()
    plt.plot(rewards_history)
    plt.title("Reward vs Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()


if __name__ == "__main__":
    train()
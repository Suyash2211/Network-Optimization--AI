import torch
from environment import TelecomEnvironment
from dqn_model import DQN


def run_demo():
    env = TelecomEnvironment()

    model = DQN(6, 4)
    model.load_state_dict(torch.load("models/dqn_model.pth"))
    model.eval()

    state = env.reset()

    total_reward = 0

    print("\n🚀 Running AI Network Optimization Demo...\n")

    for step in range(20):
        state_tensor = torch.FloatTensor(state)

        with torch.no_grad():
            q_values = model(state_tensor)
            action = torch.argmax(q_values).item()

        next_state, reward, done = env.step(action)

        print(f"Step {step+1}")
        print(f"Action Taken: {action}")
        print(f"Reward: {reward}")
        print("-" * 30)

        total_reward += reward
        state = next_state

        if done:
            break

    print(f"\n✅ Total Reward: {total_reward}")


if __name__ == "__main__":
    run_demo()
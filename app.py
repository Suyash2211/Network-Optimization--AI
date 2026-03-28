import streamlit as st
import torch
import pandas as pd
import matplotlib.pyplot as plt

from rl_agent.environment import TelecomEnvironment
from rl_agent.dqn_model import DQN


st.set_page_config(page_title="AI Network Optimization", layout="wide")

st.title("📡 AI-Based Network Optimization System")
st.caption("Built by Suyash Sinha | AI/ML Engineer")
st.write("Using ML + Reinforcement Learning to reduce congestion")

# Load environment
env = TelecomEnvironment()

# Load trained model
model = DQN(6, 4)
model.load_state_dict(torch.load("models/dqn_model.pth"))
model.eval()

# 🔥 Initialize session state
if "state" not in st.session_state:
    st.session_state.state = env.reset()
    st.session_state.total_reward = 0
    st.session_state.step = 0
    st.session_state.reward_history = []


# 🔥 Action mapping
def action_name(action):
    return {
        0: "Do Nothing",
        1: "Increase Bandwidth",
        2: "Reduce Load",
        3: "Boost Signal"
    }[action]


# 🔘 Run Single Step
if st.button("Run AI Step"):
    state_tensor = torch.FloatTensor(st.session_state.state)

    with torch.no_grad():
        q_values = model(state_tensor)
        action = torch.argmax(q_values).item()

    next_state, reward, done = env.step(action)

    st.session_state.state = next_state
    st.session_state.total_reward += reward
    st.session_state.step += 1
    st.session_state.reward_history.append(st.session_state.total_reward)

    st.success(f"Action: {action_name(action)}")
    st.info(f"Reward: {reward}")
    st.write("🧠 Q-values:", q_values.numpy())


# ⚡ Auto Run
if st.button("Auto Run (10 Steps)"):
    for _ in range(10):
        state_tensor = torch.FloatTensor(st.session_state.state)

        with torch.no_grad():
            q_values = model(state_tensor)
            action = torch.argmax(q_values).item()

        next_state, reward, done = env.step(action)

        st.session_state.state = next_state
        st.session_state.total_reward += reward
        st.session_state.step += 1
        st.session_state.reward_history.append(st.session_state.total_reward)

        if done:
            break

    st.success("Auto simulation completed!")


# 📊 Display Metrics
state = st.session_state.state

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Active Users", f"{state[0]:.2f}")
    st.metric("Bandwidth Usage", f"{state[1]:.2f}")

with col2:
    st.metric("Latency", f"{state[2]:.2f}")
    st.metric("Packet Loss", f"{state[3]:.4f}")

with col3:
    st.metric("Signal Strength", f"{state[4]:.2f}")
    st.metric("Congestion Probability", f"{state[5]:.2f}")


# 📈 Progress Section
st.subheader("📈 Progress")
st.write(f"Steps Taken: {st.session_state.step}")
st.write(f"Total Reward: {st.session_state.total_reward}")


# 📈 Reward Graph
st.subheader("📊 Reward Trend")

if len(st.session_state.reward_history) > 1:
    fig, ax = plt.subplots()
    ax.plot(st.session_state.reward_history)
    ax.set_title("Reward Over Time")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Total Reward")
    st.pyplot(fig)


# 🔄 Reset Button
if st.button("Reset Simulation"):
    st.session_state.state = env.reset()
    st.session_state.total_reward = 0
    st.session_state.step = 0
    st.session_state.reward_history = []
    st.warning("Simulation Reset")
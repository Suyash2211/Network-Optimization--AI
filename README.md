# 📡 AI-Based Network Optimization System

An end-to-end AI system that uses **Machine Learning + Reinforcement Learning** to optimize telecom network performance and reduce congestion in real time.

---

## 🚀 Project Overview

This project simulates a telecom network environment and applies AI techniques to:

- Predict network congestion
- Optimize network performance dynamically
- Learn optimal actions using Reinforcement Learning

---

## 🧠 Key Features

✔ Network traffic simulator (realistic data generation)  
✔ ML model to predict congestion (Random Forest)  
✔ RL agent (Deep Q-Network) for decision-making  
✔ Interactive Streamlit dashboard  
✔ Real-time AI decision visualization  
✔ Reward optimization from negative → positive  

---

## ⚙️ Tech Stack

- Python
- NumPy, Pandas
- Scikit-learn
- PyTorch
- Streamlit
- Matplotlib

---

## 🏗️ Project Structure
network-optimization-ai/
│
├── app.py # Streamlit UI
├── simulator/ # Data simulation
├── ml_model/ # ML model training
├── rl_agent/ # RL environment & DQN
├── data/ # Generated dataset
├── models/ # Saved ML & RL models


---

## 🔄 How It Works

1. **Simulator**
   - Generates realistic telecom traffic data

2. **Machine Learning Model**
   - Predicts congestion probability

3. **Reinforcement Learning Agent**
   - Learns actions like:
     - Increase bandwidth
     - Reduce load
     - Boost signal

4. **Streamlit UI**
   - Displays real-time decisions and performance

---

## 📈 Results

- Initial performance: Negative rewards (poor decisions)
- Final performance: Positive rewards (optimized network)

👉 Shows successful learning and optimization

---

## ▶️ How to Run

### 1. Clone Repo
```bash
git clone https://github.com/Suyash2211/Network-Optimization--AI.git
cd Network-Optimization--AI
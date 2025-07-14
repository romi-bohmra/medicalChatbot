# Medical Pre-consultation Chatbot using DQN (Final Clean Version)
# Flat 30-question action space, true DRL formulation for efficient structured data collection.

import numpy as np
import random
import torch
import json
from encoding import state_encoder
from simulator import simulated_user_response
from mapping import FEATURE_MAP
from model import DQN
import torch.nn as nn
import torch.optim as optim
from collections import deque

# -------------------- 1️⃣ PARAMETERS --------------------
NUM_ACTIONS = 30
STATE_SIZE = 30
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
LR = 1e-3
MEMORY_SIZE = 10000
TARGET_UPDATE = 10
MAX_QUESTIONS = 15
SUFFICIENCY_YES_COUNT = 3
with open('question_bank.json', 'r') as file:
    question_bank = json.load(file)

# -------------------- 3️⃣ REPLAY MEMORY --------------------
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# -------------------- 4️⃣ TRAINING LOOP --------------------
    
policy_net = DQN(STATE_SIZE, NUM_ACTIONS)
target_net = DQN(STATE_SIZE, NUM_ACTIONS)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net.to(device)
target_net.to(device)

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = deque(maxlen=MEMORY_SIZE)

epsilon = EPSILON_START
reward_history = []

for episode in range(500):
    state_vector = np.zeros(STATE_SIZE)
    state = torch.tensor(state_vector, dtype=torch.float32, device=device).unsqueeze(0)
    total_reward = 0
    questions_asked = set()

    # Simulated user profile
    user_profile = {
        "fever": 0.8,
        "cough": 0.7,
        "shortness_of_breath": 0.5,
        "headache": 0.4,
        "fatigue": 0.6,
        "abdominal_pain": 0.2,
        "diarrhea": 0.3
    }

    for step in range(MAX_QUESTIONS):
        if random.random() < epsilon:
            action = random.randrange(NUM_ACTIONS)
        else:
            with torch.no_grad():
                q_values = policy_net(state)
                q_values_np = q_values.cpu().numpy().flatten()
                q_values_np[list(questions_asked)] = -np.inf
                if np.all(q_values_np == -np.inf):
                    break
                action = int(np.argmax(q_values_np))

        user_response = simulated_user_response(action, question_bank, user_profile)
        state_np = state.squeeze(0).cpu().numpy()
        updated_state_np = state_encoder(state_np, action, user_response, question_bank)
        next_state = torch.tensor(updated_state_np, dtype=torch.float32, device=device).unsqueeze(0)

        # Reward shaping
        reward = -0.05  # mild penalty per question
        if state_np[action] > 0:
            reward -= 1  # repetition penalty
        else:
            if user_response == 'yes':
                reward += 1.5  # higher reward for informative "yes"
            else:
                reward += 0.1  # small reward for clarifying "no"

        if np.sum(updated_state_np == 2) >= SUFFICIENCY_YES_COUNT:
            reward += 5  # sufficiency bonus
            done = True
        else:
            done = False

        memory.append((state, action, next_state, torch.tensor([[reward]], device=device), done))
        state = next_state
        total_reward += reward

        if done:
            break

        if len(memory) >= BATCH_SIZE:
            batch = random.sample(memory, BATCH_SIZE)
            batch_state, batch_action, batch_next_state, batch_reward, batch_done = zip(*batch)

            batch_state = torch.cat(batch_state)
            batch_action = torch.tensor(batch_action, device=device).unsqueeze(1)
            batch_next_state = torch.cat(batch_next_state)
            batch_reward = torch.cat(batch_reward)
            batch_done_tensor = torch.tensor([not d for d in batch_done], dtype=torch.float32, device=device)

            current_q = policy_net(batch_state).gather(1, batch_action)
            next_q = target_net(batch_next_state).max(1)[0].detach()
            expected_q = batch_reward.squeeze() + GAMMA * next_q * batch_done_tensor

            loss = nn.MSELoss()(current_q.squeeze(), expected_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    reward_history.append(total_reward)
    print(f"Episode {episode} | Total Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")

# Save trained model
torch.save(policy_net.state_dict(), "trained_policy_net.pth")
print("✅ Training complete. Model saved as 'trained_policy_net.pth'.")

# Plot reward history for analysis
# import matplotlib.pyplot as plt
# plt.plot(reward_history)
# plt.xlabel('Episode')
# plt.ylabel('Total Reward')
# plt.title('Training Reward Progression')
# plt.show()
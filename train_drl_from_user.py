# train_drl_chatbot.py (Real User Data Based Training)

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from model import DQN

# Parameters
STATE_SIZE = 30
NUM_ACTIONS = 30
BATCH_SIZE = 32
GAMMA = 0.99
LR = 1e-3
NUM_EPOCHS = 10  # epochs over user data for stability
TARGET_UPDATE = 5

# Load user collected replay data
replay_data = np.load("user_replay_data.npy", allow_pickle=True)
print(f"✅ Loaded {len(replay_data)} user replay transitions.")

# Initialize networks
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(STATE_SIZE, NUM_ACTIONS).to(device)
target_net = DQN(STATE_SIZE, NUM_ACTIONS).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
criterion = nn.MSELoss()

# Prepare replay memory
memory = deque(replay_data, maxlen=len(replay_data))

# Training loop
reward_history = []
for epoch in range(NUM_EPOCHS):
    total_loss = 0
    random.shuffle(replay_data)  # shuffle each epoch for stability
    for idx in range(0, len(replay_data), BATCH_SIZE):
        batch = replay_data[idx:idx+BATCH_SIZE]
        if len(batch) < BATCH_SIZE:
            continue

        batch_state, batch_action, batch_next_state, batch_reward, batch_done = zip(*batch)

        batch_state = torch.tensor(np.stack(batch_state), dtype=torch.float32, device=device)
        batch_action = torch.tensor(batch_action, dtype=torch.long, device=device).unsqueeze(1)
        batch_next_state = torch.tensor(np.stack(batch_next_state), dtype=torch.float32, device=device)
        batch_reward = torch.tensor(batch_reward, dtype=torch.float32, device=device)
        batch_done_tensor = torch.tensor([not d for d in batch_done], dtype=torch.float32, device=device)

        current_q = policy_net(batch_state).gather(1, batch_action).squeeze()
        next_q = target_net(batch_next_state).max(1)[0].detach()
        expected_q = batch_reward + GAMMA * next_q * batch_done_tensor

        loss = criterion(current_q, expected_q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Update target network periodically
    if epoch % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    avg_loss = total_loss / (len(replay_data) / BATCH_SIZE)
    reward_history.append(avg_loss)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Avg Loss: {avg_loss:.4f}")

# Save trained policy
torch.save(policy_net.state_dict(), "trained_policy_net.pth")
print("✅ Training complete. Model saved as 'trained_policy_net.pth'.")

# Plot training loss for monitoring
# import matplotlib.pyplot as plt
# plt.plot(reward_history)
# plt.xlabel('Epoch')
# plt.ylabel('Average Loss')
# plt.title('Training Loss Progression (Real User Data)')
# plt.show()

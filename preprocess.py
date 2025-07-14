import os
import json
import numpy as np
from encoding import state_encoder
from mapping import FEATURE_MAP

# Configuration
STATE_SIZE = len(FEATURE_MAP)
SESSION_FOLDER = "user_sessions"
OUTPUT_FILE = "user_replay_data.npy"

# Container for replay data
replay_data = []

# Iterate over collected user session JSON files
for filename in os.listdir(SESSION_FOLDER):
    if filename.endswith(".json"):
        filepath = os.path.join(SESSION_FOLDER, filename)
        with open(filepath, 'r') as f:
            session = json.load(f)

        state_vector = np.zeros(STATE_SIZE)
        asked_questions = []
        chat_history = session.get("chat_history", [])

        for qa_pair in chat_history:
            question_text, user_response = qa_pair

            # Identify the action_id from the question bank
            with open('question_bank.json', 'r') as qb_file:
                question_bank = json.load(qb_file)

            action_id = None
            for q in question_bank['questions']:
                if q['text'] == question_text:
                    action_id = q['id']
                    break
            if action_id is None:
                print(f"Warning: Question text not found in bank: {question_text}")
                continue

            prev_state = state_vector.copy()
            state_vector = state_encoder(state_vector, action_id, user_response, question_bank)
            next_state = state_vector.copy()

            # Reward shaping
            reward = -0.05  # mild cost per question
            if action_id in asked_questions:
                reward -= 1  # repetition penalty
            else:
                if user_response.lower() == "yes":
                    reward += 1.5
                else:
                    reward += 0.1

            asked_questions.append(action_id)

            # Check sufficiency for early termination
            done = bool(np.sum(next_state == 2) >= 3 or len(asked_questions) >= 15)

            # Store tuple: (state, action, next_state, reward, done)
            replay_data.append((prev_state, action_id, next_state, reward, done))

# ✅ Fix: Convert to object array before saving
replay_data_array = np.array(replay_data, dtype=object)
np.save(OUTPUT_FILE, replay_data_array, allow_pickle=True)
print(f"✅ Preprocessing complete. Saved {len(replay_data)} transitions to {OUTPUT_FILE}.")
# streamlit_chatbot.py
# DRL Medical Pre-consultation Chatbot (Live Testing)

import streamlit as st
import json
import numpy as np
import torch
from encoding import state_encoder
from mapping import FEATURE_MAP
from train_drl_chatbot import DQN  # your DQN architecture file

# Load trained policy
STATE_SIZE = 30
NUM_ACTIONS = 30
policy_net = DQN(STATE_SIZE, NUM_ACTIONS)
policy_net.load_state_dict(torch.load("trained_policy_net.pth", map_location=torch.device('cpu')))
policy_net.eval()

# Load question bank
with open('question_bank.json', 'r') as f:
    question_bank = json.load(f)

st.write("✅ App loaded successfully.")

try:
    policy_net = DQN(STATE_SIZE, NUM_ACTIONS)
    policy_net.load_state_dict(torch.load("trained_policy_net.pth", map_location=torch.device('cpu')))
    policy_net.eval()
    st.write("✅ Model loaded successfully.")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")

try:
    with open('question_bank.json', 'r') as f:
        question_bank = json.load(f)
    st.write("✅ Question bank loaded successfully.")
except Exception as e:
    st.error(f"❌ Failed to load question bank: {e}")


st.title("🩺 DRL Medical Pre-consultation Chatbot")
st.write("Collect structured pre-consultation data efficiently.")

if 'state_vector' not in st.session_state:
    st.session_state.state_vector = np.zeros(STATE_SIZE)
    st.session_state.asked_questions = set()
    st.session_state.chat_history = []
    st.session_state.done = False

# Termination criteria
SUFFICIENCY_YES_COUNT = 3
MAX_QUESTIONS = 15

if not st.session_state.done:
    state = torch.tensor(st.session_state.state_vector, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = policy_net(state)
        q_values_np = q_values.detach().cpu().numpy().flatten()
        asked = list(st.session_state.asked_questions)
        q_values_np[asked] = -np.inf

        if np.all(q_values_np == -np.inf):
            st.session_state.done = True
            st.success("✅ All questions have been asked.")
            st.experimental_rerun()
        else:
            action_id = int(np.argmax(q_values_np))

    question = question_bank['questions'][action_id]
    st.write(f"**{question['text']}**")
    user_response = st.radio("Your response:", ["Yes", "No"])

    #st.write(f"Questions asked: {len(st.session_state.asked_questions)} / {MAX_QUESTIONS}")
    #st.write(f"'Yes' responses: {int(np.sum(st.session_state.state_vector == 2))} / {SUFFICIENCY_YES_COUNT}")

    if st.button("Submit Response"):
        st.session_state.asked_questions.add(action_id)
        st.session_state.chat_history.append((question['text'], user_response))
        st.session_state.state_vector = state_encoder(
            st.session_state.state_vector, action_id, user_response, question_bank
        )

        if (
            np.sum(st.session_state.state_vector == 2) >= SUFFICIENCY_YES_COUNT
            or len(st.session_state.asked_questions) >= MAX_QUESTIONS
        ):
            st.session_state.done = True

        st.rerun()

else:
    st.success("✅ Data collection completed.")
    st.write("### Your Pre-consultation Report")
    report = {}
    for feature, idx in FEATURE_MAP.items():
        val = int(st.session_state.state_vector[idx])
        if val == 2:
            report[feature] = "Yes"
        elif val == 1:
            report[feature] = "No"
        else:
            report[feature] = "Not Asked"
    st.json(report)

    st.write("### Conversation Summary")
    for q, r in st.session_state.chat_history:
        st.write(f"**Q:** {q}")
        st.write(f"**A:** {r}")

    import json
    st.download_button(
        "Download Report as JSON",
        json.dumps(report, indent=2),
        file_name="pre_consultation_report.json",
        mime="application/json"
    )

    if st.button("Reset Session"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

import json
from datetime import datetime
import os

if st.session_state.done and 'saved' not in st.session_state:
    if not os.path.exists("user_sessions"):
        os.makedirs("user_sessions")

    session_data = {
        "asked_questions": list(st.session_state.asked_questions),
        "chat_history": st.session_state.chat_history,
        "timestamp": datetime.now().isoformat()
    }

    filename = f"user_sessions/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump(session_data, f, indent=2)

    st.session_state.saved = True
    st.success(f"✅ Session saved as {filename}")


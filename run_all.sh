#!/bin/bash

# run_all.sh - Automates preprocessing, training, and launching Streamlit chatbot

echo "✅ Starting end-to-end DRL chatbot pipeline"

# Step 1: Preprocess user session data
echo "🔄 Preprocessing user session data..."
python preprocess.py

# Step 2: Train DQN using real user data
echo "🔄 Training DQN model with real user data..."
python train_drl_chatbot.py

# Step 3: Launch Streamlit chatbot for live testing
echo "🚀 Launching Streamlit chatbot..."
streamlit run chatbotUI.py

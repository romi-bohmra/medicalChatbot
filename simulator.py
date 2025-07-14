# simulator.py

import random
import json

# Load question bank
with open('question_bank.json', 'r') as file:
    question_bank = json.load(file)

def simulated_user_response(action_id, question_bank, user_profile):
    """
    Simulates a structured user response ("yes"/"no") for the chatbot's question during DRL training.

    Parameters:
    - action_id (int): The ID of the question being asked.
    - question_bank (dict): Loaded JSON question bank.
    - user_profile (dict): {feature_name: probability_yes} for simulation.

    Returns:
    - str: 'yes' or 'no'
    """
    question = question_bank['questions'][action_id]
    feature = question['feature_update']

    # Use user_profile probabilities to determine likelihood of "yes"
    probability_yes = user_profile.get(feature, 0.5)  # Default 50% chance if not defined

    response = 'yes' if random.random() < probability_yes else 'no'
    return response

# Example test:
if __name__ == "__main__":
    # Example simulated user profile:
    user_profile = {
        "fever": 0.8,
        "cough": 0.7,
        "shortness_of_breath": 0.5,
        "headache": 0.4,
        "fatigue": 0.6,
        "abdominal_pain": 0.2,
        "diarrhea": 0.3
        # Add other features as needed
    }

    for test_id in range(5):
        response = simulated_user_response(test_id, question_bank, user_profile)
        print(f"Action ID {test_id} -> Response: {response}")
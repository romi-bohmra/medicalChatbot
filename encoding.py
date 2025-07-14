# encoding.py
from mapping import FEATURE_MAP

def state_encoder(state_vector, action_id, user_response, question_bank):
    """
    Updates the state vector based on the user's structured response.

    Parameters:
    - state_vector (np.array): Current state vector (length 30).
    - action_id (int): The ID of the question asked.
    - user_response (str/int): User's structured response ('yes'/'no' or 1/0).
    - question_bank (dict): Loaded JSON question bank.

    Returns:
    - np.array: Updated state vector.
    """
    question = question_bank['questions'][action_id]
    feature_name = question['feature_update']

    # Map feature_name to index in the state vector
    feature_index = FEATURE_MAP[feature_name]

    # For your current formulation, all questions are 'binary'
    if isinstance(user_response, str):
        user_response = user_response.strip().lower()
        if user_response in ['yes', 'y', '1']:
            value = 2  # 'Yes'
        elif user_response in ['no', 'n', '0']:
            value = 1  # 'No'
        else:
            raise ValueError(f"Invalid string response: {user_response}. Expected 'yes'/'no'.")
    elif isinstance(user_response, int):
        value = 2 if user_response == 1 else 1
    else:
        raise TypeError(f"Invalid type for user_response: {type(user_response)}. Expected str or int.")

    state_vector[feature_index] = value
    return state_vector
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import pandas as pd

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

class MedicalEnvironment:
    """
    Medical consultation environment that simulates patient interactions
    """
    def __init__(self):
        # Define symptoms and their corresponding checklists
        self.symptoms = {
            'fever': ['cold', 'cough', 'sore_throat', 'travel_history', 'covid_history'],
            'chest_pain': ['shortness_of_breath', 'heart_palpitations', 'exercise_pain', 'smoking_history', 'family_heart_disease'],
            'diarrhea': ['nausea', 'vomiting', 'stomach_ache', 'travel_history', 'junk_food_consumption']
        }

        # Create question bank
        self.question_bank = [
            'cold', 'cough', 'sore_throat', 'travel_history', 'covid_history',
            'shortness_of_breath', 'heart_palpitations', 'exercise_pain', 'smoking_history',
            'family_heart_disease', 'nausea', 'vomiting', 'stomach_ache', 'junk_food_consumption',
            'headache', 'fatigue', 'dizziness', 'back_pain', 'joint_pain', 'skin_rash'
        ]

        self.state_size = len(self.question_bank)
        self.action_size = len(self.question_bank)

        # Reward structure
        self.RELEVANT_REWARD = 1.0
        self.IRRELEVANT_REWARD = -0.8
        self.QUESTION_COST = -0.2
        self.REPEATED_QUESTION_PENALTY = -1.0

        self.max_questions = 20
        self.reset()

    def reset(self):
        """Reset the environment for a new episode"""
        self.state = np.zeros(self.state_size)  # 0: Not asked, 1: No, 2: Yes
        self.asked_questions = set()
        self.current_symptom = None
        self.relevant_checklist = []
        self.questions_asked = 0
        self.episode_reward = 0

        # Simulate initial symptom selection (this would be user input)
        self.current_symptom = random.choice(list(self.symptoms.keys()))
        self.relevant_checklist = self.symptoms[self.current_symptom].copy()

        return self.state.copy()

    def step(self, action):
        """Execute action and return new state, reward, done flag"""
        question = self.question_bank[action]
        reward = self.QUESTION_COST  # Base cost for asking a question

        # Check if question was already asked
        if question in self.asked_questions:
            reward += self.REPEATED_QUESTION_PENALTY
            return self.state.copy(), reward, False, {'question': question, 'response': 'repeated'}

        # Add question to asked set
        self.asked_questions.add(question)
        self.questions_asked += 1

        # Check relevancy
        if question in self.relevant_checklist:
            reward += self.RELEVANT_REWARD
            self.relevant_checklist.remove(question)
            # Simulate user response (70% chance of yes for relevant questions)
            response = 'yes' if random.random() < 0.7 else 'no'
            self.state[action] = 2 if response == 'yes' else 1
        else:
            reward += self.IRRELEVANT_REWARD
            # Simulate user response (20% chance of yes for irrelevant questions)
            response = 'yes' if random.random() < 0.2 else 'no'
            self.state[action] = 2 if response == 'yes' else 1

        # Check termination conditions
        done = (len(self.relevant_checklist) == 0 or
                self.questions_asked >= self.max_questions)

        self.episode_reward += reward

        return self.state.copy(), reward, done, {
            'question': question,
            'response': response,
            'remaining_checklist': len(self.relevant_checklist),
            'total_reward': self.episode_reward
        }

    def get_valid_actions(self):
        """Get list of valid actions (unasked questions)"""
        return [i for i, q in enumerate(self.question_bank) if q not in self.asked_questions]

class DQNNetwork(nn.Module):
    """
    Deep Q-Network for medical chatbot
    """
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQNNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, state):
        return self.network(state)

class DQNAgent:
    """
    DQN Agent for medical consultation chatbot
    """
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.batch_size = 32
        self.update_target_freq = 100
        self.step_count = 0

        # Neural networks
        self.q_network = DQNNetwork(state_size, action_size)
        self.target_network = DQNNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Update target network
        self.update_target_network()

    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, valid_actions=None):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            # Random action from valid actions
            if valid_actions:
                return random.choice(valid_actions)
            return random.randrange(self.action_size)

        # Get Q-values for all actions
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)

        # If valid actions specified, mask invalid actions
        if valid_actions:
            masked_q_values = q_values.clone()
            invalid_actions = [i for i in range(self.action_size) if i not in valid_actions]
            masked_q_values[0, invalid_actions] = float('-inf')
            return masked_q_values.argmax().item()

        return q_values.argmax().item()

    def replay(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return 0

        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.99 * next_q_values * ~dones)

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.step_count += 1
        if self.step_count % self.update_target_freq == 0:
            self.update_target_network()

        return loss.item()

def train_agent(episodes=1000):
    """Train the DQN agent"""
    env = MedicalEnvironment()
    agent = DQNAgent(env.state_size, env.action_size)

    scores = []
    losses = []
    questions_per_episode = []
    relevant_questions_ratio = []
    epsilon_history = []

    print("Starting training...")

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        episode_losses = []
        relevant_questions = 0

        while True:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break

            action = agent.act(state, valid_actions)
            next_state, reward, done, info = env.step(action)

            # Track relevant questions
            if info['question'] in env.symptoms[env.current_symptom]:
                relevant_questions += 1

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # Train the agent
            loss = agent.replay()
            if loss > 0:
                episode_losses.append(loss)

            if done:
                break

        scores.append(total_reward)
        questions_per_episode.append(env.questions_asked)
        if env.questions_asked > 0:
            relevant_questions_ratio.append(relevant_questions / env.questions_asked)
        else:
            relevant_questions_ratio.append(0)
        epsilon_history.append(agent.epsilon)

        if episode_losses:
            losses.append(np.mean(episode_losses))
        else:
            losses.append(0)

        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            avg_questions = np.mean(questions_per_episode[-100:])
            avg_relevancy = np.mean(relevant_questions_ratio[-100:])
            print(f"Episode {episode}, Avg Score: {avg_score:.2f}, "
                  f"Avg Questions: {avg_questions:.1f}, "
                  f"Avg Relevancy: {avg_relevancy:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}")

    return agent, env, {
        'scores': scores,
        'losses': losses,
        'questions_per_episode': questions_per_episode,
        'relevant_questions_ratio': relevant_questions_ratio,
        'epsilon_history': epsilon_history
    }

def test_trained_agent(agent, env, num_tests=10):
    """Test the trained agent"""
    print("\n" + "="*50)
    print("TESTING TRAINED AGENT")
    print("="*50)

    test_results = []
    agent.epsilon = 0  # No exploration during testing

    for test in range(num_tests):
        state = env.reset()
        total_reward = 0
        questions_asked = []
        responses = []

        print(f"\nTest {test + 1}: Patient has {env.current_symptom}")
        print(f"Relevant checklist: {env.relevant_checklist}")
        print("-" * 30)

        step = 0
        while True:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break

            action = agent.act(state, valid_actions)
            next_state, reward, done, info = env.step(action)

            questions_asked.append(info['question'])
            responses.append(info['response'])

            print(f"Q{step + 1}: {info['question']} -> {info['response']} "
                  f"(Reward: {reward:.1f})")

            state = next_state
            total_reward += reward
            step += 1

            if done:
                print(f"Episode finished. Total reward: {total_reward:.2f}")
                print(f"Questions asked: {len(questions_asked)}")
                print(f"Checklist items remaining: {info['remaining_checklist']}")
                break

        test_results.append({
            'symptom': env.current_symptom,
            'total_reward': total_reward,
            'questions_count': len(questions_asked),
            'questions': questions_asked,
            'responses': responses,
            'checklist_completed': info['remaining_checklist'] == 0
        })

    return test_results

def create_visualizations(training_history):
    """Create comprehensive visualizations"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Medical Chatbot DQN Training Results', fontsize=16, fontweight='bold')

    # 1. Training Scores
    axes[0, 0].plot(training_history['scores'], alpha=0.6, color='blue')
    axes[0, 0].plot(pd.Series(training_history['scores']).rolling(100).mean(),
                    color='red', linewidth=2, label='100-episode average')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 2. Loss over time
    axes[0, 1].plot(training_history['losses'], color='orange', alpha=0.7)
    axes[0, 1].plot(pd.Series(training_history['losses']).rolling(50).mean(),
                    color='darkred', linewidth=2, label='50-episode average')
    axes[0, 1].set_title('Training Loss')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 3. Questions per episode
    axes[0, 2].plot(training_history['questions_per_episode'], color='green', alpha=0.6)
    axes[0, 2].plot(pd.Series(training_history['questions_per_episode']).rolling(100).mean(),
                    color='darkgreen', linewidth=2, label='100-episode average')
    axes[0, 2].set_title('Questions per Episode')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Number of Questions')
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    # 4. Relevancy ratio
    axes[1, 0].plot(training_history['relevant_questions_ratio'], color='purple', alpha=0.6)
    axes[1, 0].plot(pd.Series(training_history['relevant_questions_ratio']).rolling(100).mean(),
                    color='darkmagenta', linewidth=2, label='100-episode average')
    axes[1, 0].set_title('Relevant Questions Ratio')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Ratio')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 5. Epsilon decay
    axes[1, 1].plot(training_history['epsilon_history'], color='brown')
    axes[1, 1].set_title('Exploration Rate (Epsilon)')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Epsilon')
    axes[1, 1].grid(True)

    # 6. Performance distribution (last 200 episodes)
    recent_scores = training_history['scores'][-200:]
    axes[1, 2].hist(recent_scores, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
    axes[1, 2].axvline(np.mean(recent_scores), color='red', linestyle='--',
                       linewidth=2, label=f'Mean: {np.mean(recent_scores):.2f}')
    axes[1, 2].set_title('Reward Distribution (Last 200 Episodes)')
    axes[1, 2].set_xlabel('Total Reward')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def analyze_test_results(test_results):
    """Analyze and visualize test results"""
    print("\n" + "="*50)
    print("TEST RESULTS ANALYSIS")
    print("="*50)

    # Summary statistics
    avg_reward = np.mean([r['total_reward'] for r in test_results])
    avg_questions = np.mean([r['questions_count'] for r in test_results])
    completion_rate = np.mean([r['checklist_completed'] for r in test_results])

    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Questions Asked: {avg_questions:.1f}")
    print(f"Checklist Completion Rate: {completion_rate:.1%}")

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Rewards by symptom
    symptoms = [r['symptom'] for r in test_results]
    rewards = [r['total_reward'] for r in test_results]

    symptom_rewards = {}
    for s, r in zip(symptoms, rewards):
        if s not in symptom_rewards:
            symptom_rewards[s] = []
        symptom_rewards[s].append(r)

    axes[0].bar(symptom_rewards.keys(), [np.mean(v) for v in symptom_rewards.values()])
    axes[0].set_title('Average Reward by Symptom')
    axes[0].set_ylabel('Average Reward')

    # Questions count distribution
    questions_counts = [r['questions_count'] for r in test_results]
    axes[1].hist(questions_counts, bins=5, alpha=0.7, edgecolor='black')
    axes[1].set_title('Questions Asked Distribution')
    axes[1].set_xlabel('Number of Questions')
    axes[1].set_ylabel('Frequency')

    # Completion rate
    completion_data = ['Completed' if r['checklist_completed'] else 'Incomplete'
                      for r in test_results]
    completion_counts = pd.Series(completion_data).value_counts()
    axes[2].pie(completion_counts.values, labels=completion_counts.index, autopct='%1.1f%%')
    axes[2].set_title('Checklist Completion Rate')

    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Train the agent
    trained_agent, environment, training_history = train_agent(episodes=1500)

    # Create training visualizations
    create_visualizations(training_history)

    # Test the trained agent
    test_results = test_trained_agent(trained_agent, environment, num_tests=10)

    # Analyze test results
    analyze_test_results(test_results)
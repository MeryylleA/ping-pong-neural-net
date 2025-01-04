import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ReinforcementLearning:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.model = NeuralNetwork(input_size, hidden_size, output_size)
        self.target_model = NeuralNetwork(input_size, hidden_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.update_target_network()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def train(self, state, action, reward, next_state, done):
        try:
            state = torch.tensor(state, dtype=torch.float32)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            action = torch.tensor(action, dtype=torch.int64)
            reward = torch.tensor(reward, dtype=torch.float32)
            done = torch.tensor(done, dtype=torch.float32)
        except Exception as e:
            print(f"Error creating tensors: {e}")
            return

        self.memory.append((state, action, reward, next_state, done))

        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        next_states = torch.stack(next_states)
        dones = torch.stack(dones)

        try:
            q_values = self.model(states)
            next_q_values = self.target_model(next_states)
        except Exception as e:
            print(f"Error during model forward pass: {e}")
            return

        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = rewards + (1 - dones) * self.gamma * next_q_value

        loss = self.criterion(q_value, expected_q_value)

        try:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        except Exception as e:
            print(f"Error during optimizer step: {e}")
            return

    def continuous_learning(self, state, action, reward, next_state, done):
        self.train(state, action, reward, next_state, done)

    def convert_nn_output(self, output):
        return max(0, min(600 - 100, output))

    def hyperparameter_tuning(self, learning_rates, hidden_sizes, batch_sizes):
        best_params = None
        best_performance = float('-inf')

        for lr in learning rates:
            for hs in hidden_sizes:
                for bs in batch_sizes:
                    performance = self.evaluate_model(lr, hs, bs)
                    if performance > best_performance:
                        best_performance = performance
                        best_params = (lr, hs, bs)

        return best_params

    def evaluate_model(self, learning_rate, hidden_size, batch_size):
        # Placeholder for model evaluation logic
        return 0

    def adjust_learning_rate(self, optimizer, epoch, initial_lr, lr_decay):
        lr = initial_lr * (lr_decay ** epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

class AdvancedReinforcementLearning(ReinforcementLearning):
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        super().__init__(input_size, hidden_size, output_size, learning_rate)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def train(self, state, action, reward, next_state, done):
        super().train(state, action, reward, next_state, done)
        if done:
            self.update_target_network()

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.uniform(0, 600 - 100)
        else:
            with torch.no_grad():
                return self.model(torch.tensor(state, dtype=torch.float32)).item()

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def calculate_reward(self, rally_duration, precision):
        return rally_duration * precision

    def curriculum_learning(self, performance):
        if performance > 0.8:
            self.gamma = 0.95
        elif performance > 0.6:
            self.gamma = 0.9
        else:
            self.gamma = 0.85

    def visualize_training(self, state, action, reward, next_state, done):
        print(f"State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}, Done: {done}")

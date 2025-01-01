import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def train(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.int64)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        q_values = self.model(state)
        next_q_values = self.model(next_state)

        q_value = q_values.gather(0, action)
        next_q_value = next_q_values.max(0)[0]

        expected_q_value = reward + (1 - done) * next_q_value

        loss = self.criterion(q_value, expected_q_value.unsqueeze(0))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def continuous_learning(self, state, action, reward, next_state, done):
        self.train(state, action, reward, next_state, done)

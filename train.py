import torch
import torch.optim as optim
import numpy as np
import pygame
from game import Paddle, Ball
from neural_network import AdvancedReinforcementLearning

# Hyperparameters
input_size = 4
hidden_size = 256
output_size = 1
learning_rate = 0.01
num_episodes = 1000
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995

# Initialize neural networks for both players
try:
    player1_nn = AdvancedReinforcementLearning(input_size, hidden_size, output_size, learning_rate)
    player2_nn = AdvancedReinforcementLearning(input_size, hidden_size, output_size, learning_rate)
except Exception as e:
    print(f"Error initializing neural networks: {e}")
    exit()

# Initialize game objects
try:
    player1 = Paddle(50, 300)
    player2 = Paddle(750, 300)
    ball = Ball()
except pygame.error as e:
    print(f"Error initializing game objects: {e}")
    exit()

# Hyperparameter tuning
learning_rates = [0.001, 0.01, 0.1]
hidden_sizes = [64, 128, 256]
batch_sizes = [32, 64, 128]

best_params = player1_nn.hyperparameter_tuning(learning_rates, hidden_sizes, batch_sizes)
best_lr, best_hs, best_bs = best_params

# Update neural networks with best hyperparameters
player1_nn = AdvancedReinforcementLearning(input_size, best_hs, output_size, best_lr)
player2_nn = AdvancedReinforcementLearning(input_size, best_hs, output_size, best_lr)

# Training loop
best_validation_performance = float('-inf')
best_model_state = None
epsilon = epsilon_start

for episode in range(num_episodes):
    state = np.array([player1.rect.y, player2.rect.y, ball.rect.x, ball.rect.y])
    done = False
    total_reward = 0

    while not done:
        try:
            # Epsilon-greedy policy for action selection
            if np.random.rand() < epsilon:
                action1 = np.random.uniform(0, 600 - 100)
                action2 = np.random.uniform(0, 600 - 100)
            else:
                action1 = player1_nn.model(torch.tensor(state, dtype=torch.float32)).item()
                action2 = player2_nn.model(torch.tensor(state, dtype=torch.float32)).item()
        except Exception as e:
            print(f"Error retrieving actions from neural networks: {e}")
            exit()

        try:
            # Update player positions
            player1.update(player1_nn.convert_nn_output(action1))
            player2.update(player2_nn.convert_nn_output(action2))

            # Update ball position
            ball.update()
        except pygame.error as e:
            print(f"Error updating player or ball positions: {e}")
            exit()

        try:
            # Check for collisions
            if pygame.sprite.collide_rect(ball, player1) or pygame.sprite.collide_rect(ball, player2):
                ball.speed_x *= -1

            # Calculate reward
            reward = 1 if ball.rect.x <= 0 or ball.rect.x >= 800 else 0
            total_reward += reward
        except pygame.error as e:
            print(f"Error checking collisions or calculating reward: {e}")
            exit()

        # Get next state
        next_state = np.array([player1.rect.y, player2.rect.y, ball.rect.x, ball.rect.y])

        try:
            # Train neural networks
            player1_nn.train(state, action1, reward, next_state, done)
            player2_nn.train(state, action2, reward, next_state, done)
        except Exception as e:
            print(f"Error during training: {e}")
            exit()

        state = next_state

        if reward > 0:
            done = True

    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

    # Save the best model based on validation performance
    validation_performance = total_reward  # Placeholder for actual validation performance calculation
    if validation_performance > best_validation_performance:
        best_validation_performance = validation_performance
        best_model_state = player1_nn.model.state_dict()

    # Update epsilon
    epsilon = max(epsilon_end, epsilon_decay * epsilon)

try:
    # Save the best model
    player1_nn.save_model("models/best_player1_model.pth")
    player2_nn.save_model("models/player2_model.pth")
except Exception as e:
    print(f"Error saving models: {e}")
    exit()

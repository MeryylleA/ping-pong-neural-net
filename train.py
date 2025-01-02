import torch
import torch.optim as optim
import numpy as np
import pygame
from game import Paddle, Ball
from neural_network import NeuralNetwork, ReinforcementLearning

# Hyperparameters
input_size = 4
hidden_size = 128
output_size = 1
learning_rate = 0.01
num_episodes = 1000

# Initialize neural networks for both players
try:
    player1_nn = ReinforcementLearning(input_size, hidden_size, output_size, learning_rate)
    player2_nn = ReinforcementLearning(input_size, hidden_size, output_size, learning_rate)
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

# Training loop
for episode in range(num_episodes):
    state = np.array([player1.rect.y, player2.rect.y, ball.rect.x, ball.rect.y])
    done = False
    total_reward = 0

    while not done:
        try:
            # Get actions from neural networks
            action1 = player1_nn.model(torch.tensor(state, dtype=torch.float32)).item()
            action2 = player2_nn.model(torch.tensor(state, dtype=torch.float32)).item()
        except Exception as e:
            print(f"Error retrieving actions from neural networks: {e}")
            exit()

        try:
            # Update player positions
            player1.update(action1)
            player2.update(action2)

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

try:
    # Save trained models
    torch.save(player1_nn.model.state_dict(), "models/player1_model.pth")
    torch.save(player2_nn.model.state_dict(), "models/player2_model.pth")
except Exception as e:
    print(f"Error saving models: {e}")
    exit()

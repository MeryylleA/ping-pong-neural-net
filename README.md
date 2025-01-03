# Ping Pong Game with Neural Networks

This project is a ping pong game where both players are controlled by neural networks. The neural networks are implemented using PyTorch and have a continuous learning and reinforcement learning system.

## Project Purpose

The purpose of this project is to create a simple game environment where neural networks can learn and improve their performance over time through continuous learning and reinforcement learning.

## How to Run the Game

1. Clone the repository:
   ```
   git clone https://github.com/MeryylleA/ping-pong-neural-net.git
   cd ping-pong-neural-net
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the game:
   ```
   python game.py
   ```

## How to Train the Neural Networks

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Train the neural networks:
   ```
   python train.py
   ```

3. The trained models will be saved in the `models` directory.

## Hyperparameter Tuning

The training process includes hyperparameter tuning to find the optimal configuration for training the neural networks. The following hyperparameters are tuned:
- Learning rates
- Hidden layer sizes
- Batch sizes

The best model based on validation performance will be saved.

## Advanced Reinforcement Learning Techniques

The project now includes advanced reinforcement learning techniques to improve the training process. These techniques include:

- **Deep Q-Networks (DQN)**: Implemented in the `ReinforcementLearning` class to improve the training process.
- **Experience Replay**: Stores and reuses past experiences during training to help stabilize the learning process.
- **Target Networks**: Reduces the correlation between the target and predicted Q-values, improving the stability of the training process.
- **Epsilon-Greedy Policy**: Used for action selection during training to balance exploration and exploitation.
- **Enhanced Ball Physics**: The game now includes more realistic ball physics, such as spin and friction, to make the game more challenging and engaging. The ball's speed and direction are adjusted based on the angle and speed of the paddle when it hits the ball. Different ball types with varying properties, such as weight and bounce, are introduced to add variety to the gameplay.

## How to Use Trained Models in the Game

1. Ensure the trained models are saved in the `models` directory.
2. Run the game:
   ```
   python game.py
   ```

## Error Handling Improvements

The code has been updated to include error handling for various parts of the game and neural network training process. This includes error handling for:
- Pygame initialization and screen creation
- Paddle and ball creation
- Event handling in the main game loop
- Sprite updates and collision checks
- Screen updates and frame rate capping
- Tensor creation in the `train` method
- Model forward pass in the `train` method
- Loss calculation and backpropagation in the `train` method
- Optimizer step in the `train` method
- Neural network initialization
- Game object initialization
- Training loop
- Action retrieval from neural networks
- Player and ball updates
- Collision checks and reward calculation
- Model saving

## Bug Fixes and Significant Improvements

The code has also been updated to fix bugs and implement significant improvements in the game and neural network training process. These improvements include:
- Improved collision detection and handling
- Enhanced reward calculation logic
- Optimized neural network training process
- Better handling of edge cases and unexpected situations

## Scoring System

The game now includes a scoring system to keep track of points for each player. The scores are displayed on the screen during the game. When a player scores a point, the ball position is reset, and the scores are updated accordingly.

## Enhanced Ball Physics

The game now includes enhanced ball physics to make the gameplay more challenging and engaging. The following features have been added:

- **Spin and Friction**: The ball now has spin and friction, which affect its movement and interactions with the paddles and walls.
- **Angle and Speed Adjustment**: The ball's speed and direction are adjusted based on the angle and speed of the paddle when it hits the ball.
- **Different Ball Types**: The game introduces different ball types with varying properties, such as weight and bounce, to add variety to the gameplay. The ball types include:
  - **Normal Ball**: Standard ball with default properties.
  - **Heavy Ball**: Heavier ball with reduced bounce.
  - **Light Ball**: Lighter ball with increased bounce.

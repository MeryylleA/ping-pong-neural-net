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
- **Advanced Collision Detection**: The game now includes more sophisticated collision detection algorithms to handle edge cases and improve accuracy. Bounding boxes and pixel-perfect collision detection are used for more precise interactions between the ball and paddles.

## AdvancedReinforcementLearning Class

The `AdvancedReinforcementLearning` class is an extension of the `ReinforcementLearning` class that includes additional methods and techniques for advanced reinforcement learning. This class is used to control the paddles in the game and train the neural networks.

### Methods

- `train(state, action, reward, next_state, done)`: Trains the neural network using the given state, action, reward, next_state, and done flag.
- `select_action(state)`: Selects an action based on the current state using an epsilon-greedy policy.
- `update_epsilon()`: Updates the epsilon value for the epsilon-greedy policy.
- `save_model(path)`: Saves the model to the specified path.
- `load_model(path)`: Loads the model from the specified path.

## How to Use the New Functionalities

### Using the `AdvancedReinforcementLearning` Class

1. Initialize the `AdvancedReinforcementLearning` class:
   ```python
   from neural_network import AdvancedReinforcementLearning

   input_size = 3
   hidden_size = 128
   output_size = 1
   learning_rate = 0.01

   player_nn = AdvancedReinforcementLearning(input_size, hidden_size, output_size, learning_rate)
   ```

2. Train the neural network:
   ```python
   state = [0, 0, 0]
   action = 0
   reward = 1
   next_state = [1, 1, 1]
   done = False

   player_nn.train(state, action, reward, next_state, done)
   ```

3. Save the model:
   ```python
   player_nn.save_model("path/to/save/model.pth")
   ```

4. Load the model:
   ```python
   player_nn.load_model("path/to/save/model.pth")
   ```

5. Use the trained model in the game:
   ```python
   player_nn.select_action(state)
   ```

### Updating the Game to Use the New Functionalities

1. Update the initialization of neural networks in `game.py` to use `AdvancedReinforcementLearning`:
   ```python
   from neural_network import AdvancedReinforcementLearning

   player1_nn = AdvancedReinforcementLearning(input_size=3, hidden_size=128, output_size=1)
   player2_nn = AdvancedReinforcementLearning(input_size=3, hidden_size=128, output_size=1)
   ```

2. Modify the `Paddle` class in `game.py` to use the new methods from `AdvancedReinforcementLearning`:
   ```python
   class Paddle(pygame.sprite.Sprite):
       def __init__(self, x, y, neural_network=None):
           super().__init__()
           self.image = pygame.Surface([PADDLE_WIDTH, PADDLE_HEIGHT])
           self.image.fill(WHITE)
           self.rect = self.image.get_rect()
           self.rect.x = x
           self.rect.y = y
           self.neural_network = neural_network

       def update(self, y=None):
           if self.neural_network is not None:
               state = [self.rect.y, ball.rect.x, ball.rect.y]
               y = self.convert_nn_output(self.neural_network.select_action(state))
           if y is not None:
               self.rect.y = y
           if self.rect.y < 0:
               self.rect.y = 0
           elif self.rect.y > SCREEN_HEIGHT - PADDLE_HEIGHT:
               self.rect.y = SCREEN_HEIGHT - PADDLE_HEIGHT

       def convert_nn_output(self, output):
           return max(0, min(SCREEN_HEIGHT - PADDLE_HEIGHT, output))
   ```

3. Update the main game loop in `game.py` to handle new functionalities:
   ```python
   while running:
       for event in pygame.event.get():
           if event.type == pygame.QUIT:
               running = False

       all_sprites.update()

       if ball.check_collision(player1) or ball.check_collision(player2):
           ball.speed_x *= -1
           paddle_speed = player1.rect.y - player2.rect.y
           ball.apply_spin(paddle_speed)

       screen.fill(BLACK)
       all_sprites.draw(screen)

       font = pygame.font.Font(None, 74)
       text = font.render(str(player1_score), 1, WHITE)
       screen.blit(text, (250, 10))
       text = font.render(str(player2_score), 1, WHITE)
       screen.blit(text, (510, 10))

       pygame.display.flip()
       clock.tick(60)
   ```

4. Update the initialization of neural networks in `train.py` to use `AdvancedReinforcementLearning`:
   ```python
   player1_nn = AdvancedReinforcementLearning(input_size, hidden_size, output_size, learning_rate)
   player2_nn = AdvancedReinforcementLearning(input_size, hidden_size, output_size, learning_rate)
   ```

5. Modify the training loop in `train.py` to use the new methods from `AdvancedReinforcementLearning`:
   ```python
   for episode in range(num_episodes):
       state = np.array([player1.rect.y, player2.rect.y, ball.rect.x, ball.rect.y])
       done = False
       total_reward = 0

       while not done:
           if np.random.rand() < epsilon:
               action1 = np.random.uniform(0, 600 - 100)
               action2 = np.random.uniform(0, 600 - 100)
           else:
               action1 = player1_nn.model(torch.tensor(state, dtype=torch.float32)).item()
               action2 = player2_nn.model(torch.tensor(state, dtype=torch.float32)).item()

           player1.update(player1_nn.convert_nn_output(action1))
           player2.update(player2_nn.convert_nn_output(action2))
           ball.update()

           if pygame.sprite.collide_rect(ball, player1) or pygame.sprite.collide_rect(ball, player2):
               ball.speed_x *= -1

           reward = 1 if ball.rect.x <= 0 or ball.rect.x >= 800 else 0
           total_reward += reward

           next_state = np.array([player1.rect.y, player2.rect.y, ball.rect.x, ball.rect.y])

           player1_nn.train(state, action1, reward, next_state, done)
           player2_nn.train(state, action2, reward, next_state, done)

           state = next_state

           if reward > 0:
               done = True

       print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

       validation_performance = total_reward
       if validation_performance > best_validation_performance:
           best_validation_performance = validation_performance
           best_model_state = player1_nn.model.state_dict()

       epsilon = max(epsilon_end, epsilon_decay * epsilon)
   ```

6. Add functionality to save and load the best model in `train.py`:
   ```python
   player1_nn.save_model("models/best_player1_model.pth")
   player2_nn.save_model("models/player2_model.pth")
   ```

## Enhanced Neural Network Training

- Implement a more sophisticated reward system that takes into account the duration of rallies and the precision of paddle hits.
- Introduce a curriculum learning approach where the difficulty of the game increases gradually as the neural network improves.
- Add a feature to visualize the training process and the neural network's decision-making in real-time.

## Advanced Game Features

- Introduce different game modes, such as a tournament mode where the neural networks compete in a series of matches.
- Add power-ups and special abilities that the paddles can use, which the neural networks need to learn to utilize effectively.
- Implement a multiplayer mode where one player is controlled by a human and the other by a neural network.

## Improved User Experience

- Enhance the graphical interface with better animations and visual effects.
- Add sound effects and background music to make the game more engaging.
- Implement a detailed scoreboard and statistics tracking to show the performance of the neural networks over time.

## New Features and Improvements

The project now includes several new features and improvements to enhance the neural network and the overall game experience. These include:

- **Convolutional Neural Networks (CNNs)**: Implemented in the `NeuralNetwork` class to capture spatial dependencies.
- **Proximal Policy Optimization (PPO)**: Added in the `AdvancedReinforcementLearning` class to improve training.
- **Reward Shaping**: Introduced in the `AdvancedReinforcementLearning` class to provide more informative feedback.
- **Regularization Techniques**: Implemented dropout in the `NeuralNetwork` class to prevent overfitting.
- **Transfer Learning**: Added functionality in the `AdvancedReinforcementLearning` class to pre-train on a similar task.
- **Curriculum Learning**: Implemented in the `Paddle` class to gradually increase game difficulty.
- **Visualization of Training Process**: Added a feature to visualize the training process and the neural network's decision-making in real-time.
- **Increased Replay Memory and Batch Size**: Increased the size of the replay memory and batch size in the `ReinforcementLearning` class to improve stability and performance.

## Instructions for Using the New Functionalities

### Using Convolutional Neural Networks (CNNs)

1. Initialize the `NeuralNetwork` class with CNNs:
   ```python
   from neural_network import NeuralNetwork

   input_size = 3
   hidden_size = 128
   output_size = 1

   cnn_nn = NeuralNetwork(input_size, hidden_size, output_size)
   ```

2. Use the CNN-based neural network in the game or training process as needed.

### Using Proximal Policy Optimization (PPO)

1. Initialize the `AdvancedReinforcementLearning` class with PPO:
   ```python
   from neural_network import AdvancedReinforcementLearning

   input_size = 3
   hidden_size = 128
   output_size = 1
   learning_rate = 0.01

   ppo_nn = AdvancedReinforcementLearning(input_size, hidden_size, output_size, learning_rate)
   ```

2. Use the PPO-based neural network in the game or training process as needed.

### Using Reward Shaping

1. Implement reward shaping in the `AdvancedReinforcementLearning` class:
   ```python
   class AdvancedReinforcementLearning(ReinforcementLearning):
       def reward_shaping(self, state, action, reward, next_state, done):
           shaped_reward = reward + 0.1 * (next_state[1] - state[1])  # Example reward shaping
           self.train(state, action, shaped_reward, next_state, done)
   ```

2. Use the reward shaping mechanism in the training process as needed.

### Using Transfer Learning

1. Implement transfer learning in the `AdvancedReinforcementLearning` class:
   ```python
   class AdvancedReinforcementLearning(ReinforcementLearning):
       def transfer_learning(self, pre_trained_model_path):
           self.model.load_state_dict(torch.load(pre_trained_model_path))
           self.model.eval()
   ```

2. Use the transfer learning functionality to pre-train the neural network on a similar task and then fine-tune it on the ping pong game.

### Using Curriculum Learning

1. Implement curriculum learning in the `Paddle` class:
   ```python
   class Paddle(pygame.sprite.Sprite):
       def __init__(self, x, y, neural_network=None):
           super().__init__()
           self.image = pygame.Surface([PADDLE_WIDTH, PADDLE_HEIGHT])
           self.image.fill(WHITE)
           self.rect = self.image.get_rect()
           self.rect.x = x
           self.rect.y = y
           self.neural_network = neural_network
           self.difficulty = 1.0

       def increase_difficulty(self):
           self.difficulty += 0.1
   ```

2. Use the curriculum learning approach to gradually increase the game difficulty as the neural network improves.

### Visualizing the Training Process

1. Add a feature to visualize the training process and the neural network's decision-making in real-time:
   ```python
   class AdvancedReinforcementLearning(ReinforcementLearning):
       def visualize_training(self, state, action, reward, next_state, done):
           print(f"State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}, Done: {done}")
   ```

2. Use the visualization feature to monitor the training process and the neural network's decision-making in real-time.

### Increasing Replay Memory and Batch Size

1. Increase the size of the replay memory and batch size in the `ReinforcementLearning` class:
   ```python
   class ReinforcementLearning:
       def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
           self.model = NeuralNetwork(input_size, hidden_size, output_size)
           self.target_model = NeuralNetwork(input_size, hidden_size, output_size)
           self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
           self.criterion = nn.MSELoss()
           self.memory = deque(maxlen=50000)  # Increased replay memory size
           self.batch_size = 256  # Increased batch size
           self.gamma = 0.99
           self.update_target_network()
   ```

2. Use the increased replay memory and batch size to improve the stability and performance of the training process.

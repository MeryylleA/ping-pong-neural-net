# Ping Pong Game with Neural Networks

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

## Advanced Features

### Hyperparameter Tuning

The training process includes hyperparameter tuning to find the optimal configuration for training the neural networks. The following hyperparameters are tuned:
- Learning rates
- Hidden layer sizes
- Batch sizes

The best model based on validation performance will be saved.

### Advanced Reinforcement Learning Techniques

The project now includes advanced reinforcement learning techniques to improve the training process. These techniques include:

- **Deep Q-Networks (DQN)**: Implemented in the `ReinforcementLearning` class to improve the training process.
- **Experience Replay**: Stores and reuses past experiences during training to help stabilize the learning process.
- **Target Networks**: Reduces the correlation between the target and predicted Q-values, improving the stability of the training process.
- **Epsilon-Greedy Policy**: Used for action selection during training to balance exploration and exploitation.
- **Enhanced Ball Physics**: The game now includes more realistic ball physics, such as spin and friction, to make the game more challenging and engaging. The ball's speed and direction are adjusted based on the angle and speed of the paddle when it hits the ball. Different ball types with varying properties, such as weight and bounce, are introduced to add variety to the gameplay.

### Scoring System

The game now includes a scoring system to keep track of points for each player. The scores are displayed on the screen during the game. When a player scores a point, the ball position is reset, and the scores are updated accordingly.

### Enhanced Ball Physics

The game now includes enhanced ball physics to make the gameplay more challenging and engaging. The following features have been added:

- **Spin and Friction**: The ball now has spin and friction, which affect its movement and interactions with the paddles and walls.
- **Angle and Speed Adjustment**: The ball's speed and direction are adjusted based on the angle and speed of the paddle when it hits the ball.
- **Different Ball Types**: The game introduces different ball types with varying properties, such as weight and bounce, to add variety to the gameplay. The ball types include:
  - **Normal Ball**: Standard ball with default properties.
  - **Heavy Ball**: Heavier ball with reduced bounce.
  - **Light Ball**: Lighter ball with increased bounce.
- **Advanced Collision Detection**: The game now includes more sophisticated collision detection algorithms to handle edge cases and improve accuracy. Bounding boxes and pixel-perfect collision detection are used for more precise interactions between the ball and paddles.

### AdvancedReinforcementLearning Class

The `AdvancedReinforcementLearning` class is an extension of the `ReinforcementLearning` class that includes additional methods and techniques for advanced reinforcement learning. This class is used to control the paddles in the game and train the neural networks.

### Methods

- `train(state, action, reward, next_state, done)`: Trains the neural network using the given state, action, reward, next_state, and done flag.
- `select_action(state)`: Selects an action based on the current state using an epsilon-greedy policy.
- `update_epsilon()`: Updates the epsilon value for the epsilon-greedy policy.
- `save_model(path)`: Saves the model to the specified path.
- `load_model(path)`: Loads the model from the specified path.

## Error Handling

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

## Tutorials

### Modifying and Starting the System

1. To modify the game, you can edit the `game.py` file. For example, you can change the screen dimensions, paddle dimensions, ball dimensions, and colors.
2. To start the system, run the following command:
   ```
   python game.py
   ```

### Modifying and Training the Neural Network

1. To modify the neural network, you can edit the `neural_network.py` file. For example, you can change the architecture of the neural network, the learning rate, and other hyperparameters.
2. To train the neural network, run the following command:
   ```
   python train.py
   ```
3. The trained models will be saved in the `models` directory.

## Audio Device Requirement

The game requires an audio device for sound effects and background music. If no audio device is available, the game will disable sound.

### Disabling Sound

If you do not have an audio device or if you encounter issues with sound, you can disable sound by modifying the `game.py` file. Locate the following section in the code:

```python
# Check for audio device
audio_device_available = True
try:
    pygame.mixer.init()
except pygame.error as e:
    print(f"Error initializing sound: {e}")
    audio_device_available = False

# Sound effects and background music
if audio_device_available:
    try:
        hit_sound = pygame.mixer.Sound("hit.wav")
        score_sound = pygame.mixer.Sound("score.wav")
        pygame.mixer.music.load("background_music.mp3")
        pygame.mixer.music.play(-1)
    except pygame.error as e:
        print(f"Error loading sound files: {e}")
        audio_device_available = False
```

You can disable sound by setting `audio_device_available` to `False`:

```python
audio_device_available = False
```

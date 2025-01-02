import pygame
import random
import alsaaudio
import torch
from neural_network import ReinforcementLearning

# Initialize Pygame
try:
    pygame.init()
except pygame.error as e:
    print(f"Error initializing Pygame: {e}")
    exit()

def check_alsa_errors():
    try:
        alsaaudio.cards()
    except alsaaudio.ALSAAudioError as e:
        print(f"ALSA error: {e}")

check_alsa_errors()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Paddle dimensions
PADDLE_WIDTH = 10
PADDLE_HEIGHT = 100

# Ball dimensions
BALL_SIZE = 10

# Create the screen
try:
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Ping Pong Game with Neural Networks")
except pygame.error as e:
    print(f"Error creating screen: {e}")
    exit()

# Paddle class
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
            y = self.neural_network.model(torch.tensor(state, dtype=torch.float32)).item()
        if y is not None:
            self.rect.y = y
        if self.rect.y < 0:
            self.rect.y = 0
        elif self.rect.y > SCREEN_HEIGHT - PADDLE_HEIGHT:
            self.rect.y = SCREEN_HEIGHT - PADDLE_HEIGHT

# Ball class
class Ball(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface([BALL_SIZE, BALL_SIZE])
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.rect.x = SCREEN_WIDTH // 2
        self.rect.y = SCREEN_HEIGHT // 2
        self.speed_x = random.choice([-5, 5])
        self.speed_y = random.choice([-5, 5])

    def update(self):
        self.rect.x += self.speed_x
        self.rect.y += self.speed_y

        if self.rect.y <= 0 or self.rect.y >= SCREEN_HEIGHT - BALL_SIZE:
            self.speed_y *= -1

        if self.rect.x <= 0:
            self.reset_position()
            global player2_score
            player2_score += 1
        elif self.rect.x >= SCREEN_WIDTH - BALL_SIZE:
            self.reset_position()
            global player1_score
            player1_score += 1

    def reset_position(self):
        self.rect.x = SCREEN_WIDTH // 2
        self.rect.y = SCREEN_HEIGHT // 2
        self.speed_x = random.choice([-5, 5])
        self.speed_y = random.choice([-5, 5])

# Initialize neural networks for both players
player1_nn = ReinforcementLearning(input_size=3, hidden_size=128, output_size=1)
player2_nn = ReinforcementLearning(input_size=3, hidden_size=128, output_size=1)

# Create paddles and ball
try:
    player1 = Paddle(50, SCREEN_HEIGHT // 2 - PADDLE_HEIGHT // 2, neural_network=player1_nn)
    player2 = Paddle(SCREEN_WIDTH - 50 - PADDLE_WIDTH, SCREEN_HEIGHT // 2 - PADDLE_HEIGHT // 2, neural_network=player2_nn)
    ball = Ball()
except pygame.error as e:
    print(f"Error creating paddles or ball: {e}")
    exit()

# Create sprite groups
all_sprites = pygame.sprite.Group()
all_sprites.add(player1)
all_sprites.add(player2)
all_sprites.add(ball)

# Initialize scores
player1_score = 0
player2_score = 0

# Main game loop
running = True
clock = pygame.time.Clock()

while running:
    try:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    except pygame.error as e:
        print(f"Error handling events: {e}")
        running = False

    try:
        # Update game objects
        all_sprites.update()

        # Check for collisions
        if pygame.sprite.collide_rect(ball, player1) or pygame.sprite.collide_rect(ball, player2):
            ball.speed_x *= -1
    except pygame.error as e:
        print(f"Error updating sprites or checking collisions: {e}")
        running = False

    try:
        # Clear the screen
        screen.fill(BLACK)

        # Draw all sprites
        all_sprites.draw(screen)

        # Display scores
        font = pygame.font.Font(None, 74)
        text = font.render(str(player1_score), 1, WHITE)
        screen.blit(text, (250, 10))
        text = font.render(str(player2_score), 1, WHITE)
        screen.blit(text, (510, 10))

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(60)
    except pygame.error as e:
        print(f"Error updating screen or capping frame rate: {e}")
        running = False

pygame.quit()

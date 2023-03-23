from Game import Game
from Game import core
import numpy as np


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 177, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

WALL = 3
FOOD = 2
TAIL = 1
EMPTY = 0

WIDTH = 540
HEIGHT = 600
VELOCITY = 50
SHAPE = VELOCITY - 1
BOARD_COUNT = (WIDTH - 40) // VELOCITY
HOR_SHAPE = (SHAPE, SHAPE)
VER_SHAPE = (SHAPE, SHAPE)

UP = 0
DOWN = 1
RIGHT = 2
LEFT = 3


class Snake(Game):

    def __init__(self) -> None:
        super().__init__()
        self.title = "Snake"
        self.size = (WIDTH, HEIGHT)
        self.fps = 60
        self.font = core.font.SysFont("arial", 25)
        self.board = np.zeros((BOARD_COUNT, BOARD_COUNT), dtype=int)
        self.snake = []
        self.food_x = 0
        self.food_y = 0
        self.over = False
        self.action = UP
        self.frame_counter = 1
        self.frame_skip = True

    def onEvent(self, event) -> None:
        if event.type == core.KEYUP:
            if event.key == core.K_SPACE:
                self.rendering = not self.rendering
            if event.key == core.K_UP:
                if self.snake[0][2] != DOWN:
                    self.action = UP
            elif event.key == core.K_DOWN:
                if self.snake[0][2] != UP:
                    self.action = DOWN
            elif event.key == core.K_LEFT:
                if self.snake[0][2] != RIGHT:
                    self.action = LEFT
            elif event.key == core.K_RIGHT:
                if self.snake[0][2] != LEFT:
                    self.action = RIGHT

    def setup(self) -> None:
        self.reset()

    def loop(self) -> None:
        self.frame_counter += 1
        if not self.frame_skip or self.frame_counter % (VELOCITY // 2) == 0:
            self.frame_counter = 1
            if self.over:
                self.reset()
            else:
                last_dir = self.action
                for i, block in enumerate(self.snake):
                    tmp = self.snake[i][2]
                    self.snake[i][2] = last_dir
                    last_dir = tmp
                    self.move(block)
                self.food_check()

    def onRender(self) -> None:
        self.draw_game()

    def draw_game(self):
        self.window.fill((0, 0, 0))
        self.window.blit(self.font.render(f"Score: {self.score}", 1, WHITE), (230, 540))
        core.draw.line(self.window, WHITE, (20, 20), (20, 520))
        core.draw.line(self.window, WHITE, (20, 20), (520, 20))
        core.draw.line(self.window, WHITE, (20 + BOARD_COUNT * VELOCITY, 20), (20 + BOARD_COUNT * VELOCITY, 520))
        core.draw.line(self.window, WHITE, (20, 20 + BOARD_COUNT * VELOCITY), (520, 20 + BOARD_COUNT * VELOCITY))
        core.draw.rect(self.window, GREEN, (VELOCITY * self.food_y + 21, VELOCITY * self.food_x + 21, SHAPE, SHAPE))
        for i, item in enumerate(self.snake):
            if i == 0:
                if item[2] == UP or item[2] == DOWN:
                    core.draw.rect(self.window, YELLOW, (VELOCITY * item[1] + 21 + SHAPE // 2 - VER_SHAPE[0] // 2, VELOCITY * item[0] + 21 + SHAPE // 2 - VER_SHAPE[1] // 2, VER_SHAPE[0], VER_SHAPE[1]))
                elif item[2] == RIGHT or item[2] == LEFT:
                    core.draw.rect(self.window, YELLOW, (VELOCITY * item[1] + 21 + SHAPE // 2 - HOR_SHAPE[0] // 2, VELOCITY * item[0] + 21 + SHAPE // 2 - HOR_SHAPE[1] // 2, HOR_SHAPE[0], HOR_SHAPE[1]))
            else:
                if item[2] == UP or item[2] == DOWN:
                    core.draw.rect(self.window, RED, (VELOCITY * item[1] + 21 + SHAPE // 2 - VER_SHAPE[0] // 2, VELOCITY * item[0] + 21 + SHAPE // 2 - VER_SHAPE[1] // 2, VER_SHAPE[0], VER_SHAPE[1]))
                elif item[2] == RIGHT or item[2] == LEFT:
                    core.draw.rect(self.window, RED, (VELOCITY * item[1] + 21 + SHAPE // 2 - HOR_SHAPE[0] // 2, VELOCITY * item[0] + 21 + SHAPE // 2 - HOR_SHAPE[1] // 2, HOR_SHAPE[0], HOR_SHAPE[1]))

    def reset(self):
        self.over = False
        self.score = 0
        self.snake.clear()
        self.board = np.zeros((BOARD_COUNT, BOARD_COUNT), dtype=int)
        self.action = np.random.randint(0, 4)
        if self.action == DOWN:
            x = np.random.randint(2, BOARD_COUNT - 1)
            y = np.random.randint(0, BOARD_COUNT - 1)
        elif self.action == RIGHT:
            y = np.random.randint(2, BOARD_COUNT - 1)
            x = np.random.randint(0, BOARD_COUNT - 1)
        elif self.action == UP:
            x = np.random.randint(0, BOARD_COUNT - 3)
            y = np.random.randint(0, BOARD_COUNT - 1)
        elif self.action == LEFT:
            x = np.random.randint(0, BOARD_COUNT - 1)
            y = np.random.randint(0, BOARD_COUNT - 3)
        self.board[x][y] = TAIL
        self.snake.append([x, y, self.action])
        self.add_tail()
        self.add_tail()
        self.create_food()

    def create_food(self):
        while True:
            self.food_x = np.random.randint(0, BOARD_COUNT - 1)
            self.food_y = np.random.randint(0, BOARD_COUNT - 1)
            if self.board[self.food_x][self.food_y] == EMPTY:
                self.board[self.food_x][self.food_y] = FOOD
                break

    def food_check(self):
        if self.snake[0][0] == self.food_x and self.snake[0][1] == self.food_y:
            self.score += 1
            self.add_tail()
            self.create_food()

    def move(self, block):
        x = block[0]
        y = block[1]
        dir = block[2]
        if dir == UP:
            if x == 0 or self.board[x - 1][y] == TAIL:
                self.over = True
            else:
                self.board[x - 1][y] = TAIL
                block[0] -= 1
        elif dir == DOWN:
            if x == BOARD_COUNT - 1 or self.board[x + 1][y] == TAIL:
                self.over = True
            else:
                self.board[x + 1][y] = TAIL
                block[0] += 1
        elif dir == RIGHT:
            if y == BOARD_COUNT - 1 or self.board[x][y + 1] == TAIL:
                self.over = True
            else:
                self.board[x][y + 1] = TAIL
                block[1] += 1
        elif dir == LEFT:
            if y == 0 or self.board[x][y - 1] == TAIL:
                self.over = True
            else:
                self.board[x][y - 1] = TAIL
                block[1] -= 1
        self.board[x][y] = EMPTY

    def add_tail(self):
        tail = self.snake[-1].copy()
        if tail[2] == UP:
            tail[0] += 1
        elif tail[2] == DOWN:
            tail[0] -= 1
        elif tail[2] == LEFT:
            tail[1] += 1
        elif tail[2] == RIGHT:
            tail[1] -= 1
        if self.board[tail[0], tail[1]] != EMPTY:
            print("Can't add tail")
            print(self.snake[0])
            print(tail)
            print(self.food_x, self.food_y)
            print(self.board)
            self.running = False
        else:
            self.board[tail[0]][tail[1]] = TAIL
            self.snake.append(tail)

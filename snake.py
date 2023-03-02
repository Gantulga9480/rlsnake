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
VELOCITY = 60
SHAPE = VELOCITY - 1
BOARD_COUNT = int((WIDTH - 40) / VELOCITY)
HOR_SHAPE = (SHAPE, SHAPE)
VER_SHAPE = (SHAPE, SHAPE)
UP = 0
DOWN = 1
RIGHT = 2
LEFT = 3
FORWARD = 0
TLEFT = 1
TRIGHT = 2
ACTION_SPACE = [0, 1, 2, 3]
ACTION_SPACE_SIZE = 4
STATE_SPACE_SIZE = 4
FOOD_REWARD = 1
OUT_REWARD = 0
EMPTY_STEP_REWARD = 0


class Snake(Game):

    def __init__(self) -> None:
        super().__init__()
        self.size = (WIDTH, HEIGHT)
        self.fps = 60
        self.font = core.font.SysFont("arial", 25)
        self.board = np.zeros((BOARD_COUNT, BOARD_COUNT), dtype=int)
        self.snake = []
        self.food_x = 0
        self.food_y = 0
        self.over = False
        self.food_hit = False
        self.set_title("Snake")
        self.set_window()

    def onEvent(self, event) -> None:
        if event.type == core.KEYUP:
            if event.key == core.K_SPACE:
                self.rendering = not self.rendering

    def loop(self) -> None:
        ...

    def loop_once(self) -> bool:
        super().loop_once()
        return True if not self.running else self.over

    def onRender(self) -> None:
        self.draw_game()

    def step(self, action):
        last_dir = action
        for i, block in enumerate(self.snake):
            tmp = self.snake[i][2]
            self.snake[i][2] = last_dir
            last_dir = tmp
            self.move(block)
        self.food_check()
        return self.feedback()

    def feedback(self):
        if self.over:
            return self.get_state(), OUT_REWARD, self.over
        elif self.food_hit:
            self.food_hit = False
            return self.get_state(), FOOD_REWARD, self.over
        else:
            return self.get_state(), EMPTY_STEP_REWARD, self.over

    def get_state(self):
        state = []
        x = self.snake[0][0]
        y = self.snake[0][1]
        d = self.snake[0][2]
        if d == UP:
            state.append(self.board[x][y - 1]) if y > 0 else state.append(WALL)
            state.append(self.board[x - 1][y]) if x > 0 else state.append(WALL)
            state.append(self.board[x][y + 1]) if y < BOARD_COUNT - 1 else state.append(WALL)
        elif d == RIGHT:
            state.append(self.board[x - 1][y]) if x > 0 else state.append(WALL)
            state.append(self.board[x][y + 1]) if y < BOARD_COUNT - 1 else state.append(WALL)
            state.append(self.board[x + 1][y]) if x < BOARD_COUNT - 1 else state.append(WALL)
        elif d == DOWN:
            state.append(self.board[x][y + 1]) if y < BOARD_COUNT - 1 else state.append(WALL)
            state.append(self.board[x + 1][y]) if x < BOARD_COUNT - 1 else state.append(WALL)
            state.append(self.board[x][y - 1]) if y > 0 else state.append(WALL)
        elif d == LEFT:
            state.append(self.board[x + 1][y]) if x < BOARD_COUNT - 1 else state.append(WALL)
            state.append(self.board[x][y - 1]) if y > 0 else state.append(WALL)
            state.append(self.board[x - 1][y]) if x > 0 else state.append(WALL)
        dif_x = self.food_x - self.snake[0][0]
        dif_y = self.food_y - self.snake[0][1]
        if dif_y == 0 and dif_x < 0:
            state.append(0)
        elif dif_y > 0 and dif_x < 0:
            state.append(1)
        elif dif_y > 0 and dif_x == 0:
            state.append(2)
        elif dif_y > 0 and dif_x > 0:
            state.append(3)
        elif dif_y == 0 and dif_x > 0:
            state.append(4)
        elif dif_y < 0 and dif_x > 0:
            state.append(5)
        elif dif_y < 0 and dif_x == 0:
            state.append(6)
        elif dif_y < 0 and dif_x < 0:
            state.append(7)
        return tuple(state)

    def draw_game(self):
        self.score = len(self.snake) - 3
        self.window.fill((0, 0, 0))
        core.draw.line(self.window, WHITE, (20, 20), (20, 520))
        core.draw.line(self.window, WHITE, (20 + BOARD_COUNT * VELOCITY, 20), (20 + BOARD_COUNT * VELOCITY, 520))
        core.draw.line(self.window, WHITE, (20, 20), (520, 20))
        core.draw.line(self.window, WHITE, (20, 20 + BOARD_COUNT * VELOCITY), (520, 20 + BOARD_COUNT * VELOCITY))
        score_str = self.font.render(f"Q-Learning Score: {self.score}", 1, WHITE)
        self.window.blit(score_str, (180, 540))
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
        core.draw.rect(self.window, GREEN, (VELOCITY * self.food_y + 21, VELOCITY * self.food_x + 21, SHAPE, SHAPE))

    def reset(self):
        self.over = False
        self.snake.clear()
        self.score = 0
        self.board = np.zeros((BOARD_COUNT, BOARD_COUNT), dtype=int)
        d = np.random.randint(0, 4)
        if d == DOWN:
            x = np.random.randint(2, BOARD_COUNT - 1)
            x_1 = x - 1
            x_2 = x_1 - 1
            y = np.random.randint(0, BOARD_COUNT - 1)
            y_1 = y
            y_2 = y_1
        elif d == RIGHT:
            y = np.random.randint(2, BOARD_COUNT - 1)
            y_1 = y - 1
            y_2 = y_1 - 1
            x = np.random.randint(0, BOARD_COUNT - 1)
            x_1 = x
            x_2 = x_1
        elif d == UP:
            x = np.random.randint(0, BOARD_COUNT - 3)
            x_1 = x + 1
            x_2 = x_1 + 1
            y = np.random.randint(0, BOARD_COUNT - 1)
            y_1 = y
            y_2 = y_1
        elif d == LEFT:
            x = np.random.randint(0, BOARD_COUNT - 1)
            x_1 = x
            x_2 = x_1
            y = np.random.randint(0, BOARD_COUNT - 3)
            y_1 = y + 1
            y_2 = y_1 + 1
        self.board[x][y] = TAIL
        self.board[x_1][y_1] = TAIL
        self.board[x_2][y_2] = TAIL
        self.snake.append([x, y, d])
        self.snake.append([x_1, y_1, d])
        self.snake.append([x_2, y_2, d])
        self.create_food()
        return self.get_state()

    def create_food(self):
        while True:
            self.food_x = np.random.randint(0, BOARD_COUNT - 1)
            self.food_y = np.random.randint(0, BOARD_COUNT - 1)
            if self.board[self.food_x][self.food_y] == EMPTY:
                self.board[self.food_x][self.food_y] = FOOD
                break

    def food_check(self):
        if self.snake[0][0] == self.food_x and self.snake[0][1] == self.food_y:
            self.food_hit = True
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

    def translate_action(self, action):
        if action == FORWARD:
            return self.snake[0][2]
        elif action == TLEFT:
            if self.snake[0][2] == UP:
                return LEFT
            elif self.snake[0][2] == RIGHT:
                return UP
            elif self.snake[0][2] == DOWN:
                return RIGHT
            elif self.snake[0][2] == LEFT:
                return DOWN
        elif action == TRIGHT:
            if self.snake[0][2] == UP:
                return RIGHT
            elif self.snake[0][2] == RIGHT:
                return DOWN
            elif self.snake[0][2] == DOWN:
                return LEFT
            elif self.snake[0][2] == LEFT:
                return UP

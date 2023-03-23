from snake import core, Snake, WALL, BOARD_COUNT, UP, DOWN, LEFT, RIGHT
import numpy as np

FORWARD = 0
TLEFT = 1
TRIGHT = 2

ACTION_SPACE_SIZE = 3
STATE_SPACE_SIZE = (4, 4, 4, 8)  # Total size ?

FOUND_FOOD_REWARD = 1
GAME_OVER_REWARD = -1
EMPTY_STEP_REWARD = 0


class RL_Snake(Snake):

    def __init__(self) -> None:
        super().__init__()
        self.frame_skip = False
        self.set_title("Snake")
        self.set_window()

    def reset(self):
        """Overriding reset function to return initial game state"""
        super().reset()
        return self.state()

    def step(self, action):
        self.action = action
        last_score = self.score
        self.loop_once()
        reward = EMPTY_STEP_REWARD
        if self.over:
            reward = GAME_OVER_REWARD
        elif self.score != last_score:
            reward = FOUND_FOOD_REWARD
        return self.state(), reward, self.over

    def state(self):
        """
        Utility function to extract current state from game board

        @return 3 node (0 to 3) around snake head + food diraction (0 to 8)
        """
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

    def translate_action(self, action):
        """Utility function convert model action to game action"""
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

    def onEvent(self, event) -> None:
        """Overriding event handler to block user interaction"""
        if event.type == core.KEYUP:
            if event.key == core.K_SPACE:
                self.rendering = not self.rendering

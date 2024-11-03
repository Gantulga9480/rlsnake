from pysnake import Snake, UP, DOWN, LEFT, RIGHT, TAIL

FORWARD = 0
TLEFT = 1
TRIGHT = 2

ACTION_SPACE_SIZE = 3
STATE_SPACE_SIZE = (3, 3, 3, 3, 3)  # Total size ?


class rl_Snake(Snake):

  def __init__(self, board_size: int = 10, deep: bool = False):
    super().__init__(board_size)
    self.deep = deep

  def reset(self):
    """
    Overriding reset function to return initial game state
    """
    super().reset()
    return self.state()

  def step(self, action):
    self.head_dir = self.translate_action(action)
    last_score = self.score
    self.move()
    reward = 0
    if self.game_over:
      reward = -100
    else:
      if self.score - last_score > 0:
        reward = 10
      else:
        reward = -0.1  # reward for surviving
    # reward = -1 if self.game_over else self.score - last_score
    return self.state(), reward, self.game_over

  def state(self) -> tuple:
    """
    Utility function to extract current state from game board
    """

    if self.deep:
      return self.board.flatten()

    x = self.body[0][0]  # head x
    y = self.body[0][1]  # head y
    d = self.body[0][2]  # head dir

    state = []

    def append_state(x, y):
      if 0 <= y < self.board_size and 0 <= x < self.board_size:
        state.append(self.board[y][x])
      else:
        state.append(TAIL)

    def append_direction(value):
      if value < 0:
        state.append(-1)
      elif value == 0:
        state.append(0)
      else:
        state.append(1)

    if d == UP:
      append_state(y, x - 1)
      append_state(y - 1, x)
      append_state(y, x + 1)
    elif d == RIGHT:
      append_state(y - 1, x)
      append_state(y, x + 1)
      append_state(y + 1, x)
    elif d == DOWN:
      append_state(y, x + 1)
      append_state(y + 1, x)
      append_state(y, x - 1)
    elif d == LEFT:
      append_state(y + 1, x)
      append_state(y, x - 1)
      append_state(y - 1, x)

    append_direction(self.food_position[0] - x)
    append_direction(self.food_position[1] - y)

    return tuple(state)

  def translate_action(self, action):
    """
    Utility function convert model action (turn left/right, go forward) to game action (up, down, left, right)
    """
    if action == FORWARD:
        return self.body[0][2]
    elif action == TLEFT:
        if self.body[0][2] == UP:
            return LEFT
        elif self.body[0][2] == RIGHT:
            return UP
        elif self.body[0][2] == DOWN:
            return RIGHT
        elif self.body[0][2] == LEFT:
            return DOWN
    elif action == TRIGHT:
        if self.body[0][2] == UP:
            return RIGHT
        elif self.body[0][2] == RIGHT:
            return DOWN
        elif self.body[0][2] == DOWN:
            return LEFT
        elif self.body[0][2] == LEFT:
            return UP

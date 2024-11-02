from pysnake import Snake, UP, DOWN, LEFT, RIGHT, TAIL

FORWARD = 0
TLEFT = 1
TRIGHT = 2

ACTION_SPACE_SIZE = 3
STATE_SPACE_SIZE = (3, 3, 3, 3, 3)  # Total size ?


class rl_Snake(Snake):

  def __init__(self, board_size = 10):
    super().__init__(board_size)

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
    # reward = 0
    # if self.score - last_score > 0:
    #    reward = 400
    # if self.game_over:
    #    reward = -4000
    reward = -1 if self.game_over else self.score - last_score
    return self.state(), reward, self.game_over

  def state(self) -> tuple:
    """
    Utility function to extract current state from game board
    @return 3 node (0 to 3) around snake head, food diraction from snake head
    """
    state = []
    x = self.body[0][0]
    y = self.body[0][1]
    d = self.body[0][2]
    if d == UP:
      state.append(self.board[y][x - 1]) if x > 0 else state.append(TAIL)
      state.append(self.board[y - 1][x]) if y > 0 else state.append(TAIL)
      state.append(self.board[y][x + 1]) if x < self.board_size - 1 else state.append(TAIL)
    elif d == RIGHT:
      state.append(self.board[y - 1][x]) if y > 0 else state.append(TAIL)
      state.append(self.board[y][x + 1]) if x < self.board_size - 1 else state.append(TAIL)
      state.append(self.board[y + 1][x]) if y < self.board_size - 1 else state.append(TAIL)
    elif d == DOWN:
      state.append(self.board[y][x + 1]) if x < self.board_size - 1 else state.append(TAIL)
      state.append(self.board[y + 1][x]) if y < self.board_size - 1 else state.append(TAIL)
      state.append(self.board[y][x - 1]) if x > 0 else state.append(TAIL)
    elif d == LEFT:
      state.append(self.board[y + 1][x]) if y < self.board_size - 1 else state.append(TAIL)
      state.append(self.board[y][x - 1]) if x > 0 else state.append(TAIL)
      state.append(self.board[y - 1][x]) if y > 0 else state.append(TAIL)
    dif_x = self.food_position[0] - x
    dif_y = self.food_position[1] - y
    if dif_y < 0:
       state.append(-1)
    if dif_y == 0:
       state.append(0)
    if dif_y > 0:
       state.append(1)
    if dif_x < 0:
       state.append(-1)
    if dif_x == 0:
       state.append(0)
    if dif_x > 0:
       state.append(1)
    return tuple(state)

  def translate_action(self, action):
    """
    Utility function convert model action to game action
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

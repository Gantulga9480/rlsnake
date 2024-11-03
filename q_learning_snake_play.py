from pysnake import PySnake
from rl_snake import rl_Snake, STATE_SPACE_SIZE, ACTION_SPACE_SIZE
from pyrl.q import QLearningAgent
import pygame as pg

class QPySnake(PySnake):

  def __init__(self, board_size = 20, game_speed = 5):
    super().__init__(board_size, game_speed)
    self.snake = rl_Snake(board_size)
    self.fps = 60
    self.custom_setup()

  def onEvent(self, event):
    if event.type == pg.KEYUP:
      if event.key == pg.K_r:
        self.reset()
    super().onEvent(event)

  def reset(self):
    return self.snake.reset()

  def step(self, action):
    self.loop_once()
    return self.snake.step(action)

  def loop(self):
    pass

from tkinter import Tk, filedialog
tk_root = Tk()
tk_root.withdraw()
dir_name = filedialog.askopenfilename()

if not dir_name.endswith('.npy'):
  raise Exception(f"invalid file type {dir_name.split('.')[-1]}")

game = QPySnake(20)
agent = QLearningAgent(STATE_SPACE_SIZE, ACTION_SPACE_SIZE)
agent.load_model(dir_name)
agent.eval()

while game.running:
  state = game.reset()
  while not game.snake.game_over and game.running:
    action = agent.policy(state)
    next_state, reward, done = game.step(action)
    state = next_state

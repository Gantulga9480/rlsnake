from rl_snake import rl_Snake, STATE_SPACE_SIZE, ACTION_SPACE_SIZE
from pyrl.dqn import DeepQNetworkAgent
from pyrl.utils import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn


class DQN(nn.Module):

  def __init__(self, observation_size, action_size):
    super().__init__()
    self.model = nn.Sequential(
      nn.Linear(observation_size, 128),
      nn.LeakyReLU(),
      nn.Linear(128, 64),
      nn.LeakyReLU(),
      nn.Linear(64, 32),
      nn.LeakyReLU(),
      nn.Linear(32, action_size),
      nn.Softmax(dim=1)
    )

  def forward(self, x):
    return self.model(x)


game = rl_Snake(20, deep=True)
agent = DeepQNetworkAgent(game.board_size ** 2, ACTION_SPACE_SIZE, device='cuda')
agent.create_model(DQN, lr=0.01, gamma=0.99, e_decay=0.9995)
agent.create_buffer(ReplayBuffer(1_000_000, 10_000, game.board_size ** 2))

writer = SummaryWriter()

for i in range(10000):
  state = game.reset()
  while not game.game_over:
    action = agent.policy(state)
    next_state, reward, done = game.step(action)
    agent.learn(state, action, next_state, reward, done)
    state = next_state
  writer.add_scalars('Metric', {'Reward': agent.reward_history[-1]}, i)
  if (i + 1) % 500 == 0:
    agent.save_model(f'model_{i + 1}.pt', as_jit=True)

agent.save_model('model.pt')
writer.close()
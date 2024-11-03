from rl_snake import rl_Snake, STATE_SPACE_SIZE, ACTION_SPACE_SIZE
from pyrl.q import QLearningAgent
from torch.utils.tensorboard import SummaryWriter


game = rl_Snake(20)
agent = QLearningAgent(STATE_SPACE_SIZE, ACTION_SPACE_SIZE)
agent.create_model(lr=0.1, gamma=0.9, e_decay=0.999999)

writer = SummaryWriter()

for i in range(200000):
  state = game.reset()
  while not game.game_over:
    action = agent.policy(state)
    next_state, reward, done = game.step(action)
    agent.learn(state, action, next_state, reward, done)
    state = next_state
  writer.add_scalars('Metric', {'Reward': agent.reward_history[-1]}, i)

agent.save_model('model.npy')
writer.close()
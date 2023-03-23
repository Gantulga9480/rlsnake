from RL_Snake import RL_Snake, STATE_SPACE_SIZE, ACTION_SPACE_SIZE
from RL.q import QLearningAgent
import matplotlib.pylab as plt
import numpy as np
np.random.seed(3407)

game = RL_Snake()
agent = QLearningAgent(STATE_SPACE_SIZE, ACTION_SPACE_SIZE)
agent.create_model(lr=0.1, y=0.99, e_decay=0.999)

for _ in range(2000):
    state = game.reset()
    while not game.over and game.running:
        action = agent.policy(state)
        next_state, reward, done = game.step(game.translate_action(action))
        agent.learn(state, action, next_state, reward, done)
        state = next_state

print(f"mean: {np.mean(agent.reward_history)}")
print(f"max: {np.max(agent.reward_history)}")
plt.plot(agent.reward_history)
plt.show()

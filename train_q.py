from snake import Snake
from RL import QLearningAgent
import matplotlib.pylab as plt

game = Snake()
agent = QLearningAgent((4, 4, 4, 8), 3)
agent.create_model(0.1, 0.99, e_decay=0.9999)

scores = []
while game.running:
    reward = []
    s = game.reset()
    while not game.loop_once():
        a = agent.policy(s)
        ns, r, d = game.step(game.translate_action(a))
        agent.learn(s, a, ns, r, d)
        s = ns
        reward.append(r)
    scores.append(sum(reward))
    print(agent.e, agent.episode_count)
agent.save_model('model.npy')
plt.plot(scores)
plt.show()

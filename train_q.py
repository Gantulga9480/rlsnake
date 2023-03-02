from snake import Snake
from RL import QLAgent

game = Snake()
agent = QLAgent(0.5, 0.9)
agent.create_model((4, 4, 4, 8, 3))

use_eps = True

while game.running:
    s = game.reset()
    while not game.loop_once():
        action = agent.policy(s, greedy=use_eps)
        ns, r, d = game.step(game.translate_action(action))
        agent.learn(s, action, r, ns, d)
        agent.decay_epsilon(0.9999)
        s = ns
        print(agent.e, agent.episode_count)
agent.save_model('model.npy')
agent.plot()

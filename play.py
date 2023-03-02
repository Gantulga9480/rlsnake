from snake import Snake
from RL import QLAgent

game = Snake()
agent = QLAgent(3, 0, 0)
agent.load_model('model_test.npy')

episode = 0

while game.running:
    episode += 1
    s = game.reset()
    while not game.loop_once():
        action = agent.policy(s, greedy=True)
        s, r, d = game.step(game.translate_action(action))

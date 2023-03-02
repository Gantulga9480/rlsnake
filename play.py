from snake import Snake
from RL import QLearningAgent

game = Snake()
agent = QLearningAgent(4, 3)
agent.load_model('model_test.npy')
agent.train = False

while game.running:
    s = game.reset()
    while not game.loop_once():
        action = agent.policy(s)
        s, r, d = game.step(game.translate_action(action))

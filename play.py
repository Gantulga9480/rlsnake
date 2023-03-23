from snake import Snake
from RL_Snake import RL_Snake
from RL.q import QLearningAgent
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', "--model", type=str, help="Learned model (.npy file)")
args = parser.parse_args()

if args.model:
    game = RL_Snake()
    agent = QLearningAgent((4, 4, 4, 8), 3)
    agent.load_model(args.model)
    agent.training = False
    while game.running:
        s = game.reset()
        while not game.over and game.running:
            action = agent.policy(s)
            s, r, d = game.step(game.translate_action(action))
else:
    game = Snake()
    game.loop_forever()

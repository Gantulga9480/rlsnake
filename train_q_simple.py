from RL_Snake import RL_Snake, STATE_SPACE_SIZE, ACTION_SPACE_SIZE
from RL.q import QLearningAgent


game = RL_Snake()
agent = QLearningAgent(STATE_SPACE_SIZE, ACTION_SPACE_SIZE)
agent.create_model(lr=0.1, y=0.99, e_decay=0.999)

while game.running:
    state = game.reset()
    while not game.over and game.running:
        action = agent.policy(state)
        next_state, reward, done = game.step(game.translate_action(action))
        agent.learn(state, action, next_state, reward, done)
        state = next_state

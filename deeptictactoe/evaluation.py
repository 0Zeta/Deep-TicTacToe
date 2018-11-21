from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range

from deeptictactoe.game import Game, Randy
from deeptictactoe.q_approximator import NeuralNetwork
from deeptictactoe.rl_agent import QAgent

wins = 0
q = QAgent(approximator=NeuralNetwork(checkpoint=40), exploration_rate=0.)
r = Randy()

# evaluate the agent by letting it play against itself
for i in range(1000):
    if i % 100 == 0:
        print("Simulating round {}".format(i))
    game = Game(q, q)
    result = game.play()['result']
    if result == 1:
        wins += 1
print("\nWins Q-agent vs Q-agent: {}".format(wins))  # 0 wins -> Each game is drawn -> The Q-agent mastered the game

# let the agent play against Randy
wins = 0
for i in range(1000):
    if i % 100 == 0:
        print("Simulating round {}".format(i))
    game = Game(q, r)
    result = game.play()['result']
    if result == 1:
        wins += 1
print("\nWins Q-agent vs Randy: {}".format(wins))

wins = 0
for i in range(1000):
    if i % 100 == 0:
        print("Simulating round {}".format(i))
    game = Game(r, q)
    result = game.play()['result']
    if result == 1:
        wins += 1
print("\nWins Randy vs Q-agent: {}".format(wins))

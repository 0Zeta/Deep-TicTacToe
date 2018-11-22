from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from random import randrange, random

from six.moves import range

from deeptictactoe.game import Game
from deeptictactoe.q_approximator import NeuralNetwork
from deeptictactoe.rl_agent import QAgent

NB_EPOCHS = 51  # train NB_EPOCHS times
GAMES_PER_EPOCH = 2000  # simulate GAMES_PER_EPOCH games per epoch
CHECKPOINT = 30  # last checkpoint

if __name__ == "__main__":
    nn = NeuralNetwork(checkpoint=CHECKPOINT)
    player_1 = QAgent(nn, exploration_rate=0.05)
    player_2 = QAgent(nn, exploration_rate=0.5)
    for epoch in range(CHECKPOINT + 1, NB_EPOCHS):
        print("[*] Training epoch {}".format(epoch))
        data = []  # the game histories of the simulated games
        for i in range(GAMES_PER_EPOCH):
            if i % (GAMES_PER_EPOCH / 10) == 0:
                print("[*] Simulating game {}/{}".format(i + 1, GAMES_PER_EPOCH))
                if epoch > 2:
                    # let the Q-agent play against an older version of itself
                    player_2 = QAgent(NeuralNetwork(checkpoint=randrange(2, epoch, 2)), exploration_rate=random())
            if i % 2 == 0:  # switch sides
                game = Game(player_1, player_2)  # simulate a game with the Q-agent playing against itself
            else:
                game = Game(player_2, player_1)
            history = game.play()
            data.append(history)
        print("[*] Starting training")
        nn.train_on_history(data, epoch % 2 == 0, epoch)

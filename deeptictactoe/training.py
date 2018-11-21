from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range

from deeptictactoe.game import Game
from deeptictactoe.q_approximator import NeuralNetwork
from deeptictactoe.rl_agent import QAgent

NB_EPOCHS = 41  # train NB_EPOCHS times
GAMES_PER_EPOCH = 500  # simulate GAMES_PER_EPOCH games per epoch

if __name__ == "__main__":
    nn = NeuralNetwork()
    player1 = QAgent(nn, exploration_rate=0.3)
    player2 = QAgent(nn, exploration_rate=0.3)
    for epoch in range(NB_EPOCHS):
        print("[*] Training epoch {}".format(epoch))
        data = []  # the game histories of the simulated games
        for i in range(GAMES_PER_EPOCH):
            if i % (GAMES_PER_EPOCH / 10) == 0:
                print("[*] Simulating game {}/{}".format(i + 1, GAMES_PER_EPOCH))
            game = Game(player1, player2)
            history = game.play()
            data.append(history)
        print("[*] Starting training")
        nn.train_on_history(data, epoch % 5 == 0, epoch)

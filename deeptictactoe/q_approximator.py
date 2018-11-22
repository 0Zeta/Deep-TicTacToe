from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from keras import Sequential
from keras.layers import Dense
from six.moves import range


class QApproximator:
    """abstract class for various ways to approximate the quality function"""

    def Q(self, state, action):
        """determines the Q-value of a specific action in a specific state

        :param state: the game state
        :param action: the field which could be marked as an action

        :type state: list[int]
        :type action: (int, int)

        :returns: the estimated Q-value of the action in the specified state
        :rtype: float
        """
        raise NotImplementedError()


class NeuralNetwork:
    """This class implements an approximator for the quality function using a neural network."""

    GAMMA = 0.99  # discount factor for Q-learning
    BATCH_SIZE = 128
    NB_EPOCHS = 5

    def __init__(self, checkpoint=0):
        """initializes the neural network

        :param checkpoint: tells the program to load weights from the specified checkpoint
        :type checkpoint: int
        """
        # construct a simple feedforward-network
        model = Sequential()
        model.add(Dense(18, input_dim=18, activation='relu', use_bias=True))
        model.add(Dense(10, activation='relu', use_bias=True))
        model.add(Dense(1, use_bias=True))
        model.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])
        # load checkpoint
        if checkpoint > 0:
            model.load_weights('checkpoints/checkpoint_{}.h5py'.format(checkpoint))
        self.model = model

    def Q(self, state, action):
        x = NeuralNetwork.to_vector(state, action)
        x = np.reshape(x, [1, 18])
        return self.model.predict(x)[0]

    def train_on_history(self, data, save_weights=False, nb_checkpoint=0):
        """updates the neural network using the data from several games

        :param data: data from multiple games consisting of states, actions and results
        :param save_weights: whether to save the model after the training as a new checkpoint
        :param nb_checkpoint: the number for the new checkpoint
        :type data: list[dict]
        :type save_weights: bool
        :type nb_checkpoint: int
        """
        X = []
        y = []

        for game_history in data:
            if game_history['result'] == 0:  # check for a draw
                for player in range(1, 3):
                    # set 0 as a reward for the last move of both players
                    X.append(
                        NeuralNetwork.to_vector(game_history[str(player)][-1][0], game_history[str(player)][-1][2]))
                    y.append(0)
            else:
                for player in range(1, 3):
                    # set 1 as a reward for the last move of the winner and -1 for the loser
                    X.append(
                        NeuralNetwork.to_vector(game_history[str(player)][-1][0], game_history[str(player)][-1][2]))
                    y.append(1 if game_history['result'] == player else -1)

                    # add the rewards for the other actions as s_t = reward (0) + GAMMA * max_a_Q(s_t+1, a)
                    for i, turn in enumerate(game_history[str(player)][:-1]):
                        X.append(NeuralNetwork.to_vector(turn[0], turn[2]))
                        next_state = game_history[str(player)][i + 1][0]
                        available_actions_next_state = game_history[str(player)][i + 1][1]
                        encoded_actions = [NeuralNetwork.to_vector(next_state, action) for action in
                                           available_actions_next_state]
                        estimated_q_values = self.model.predict(np.array(encoded_actions))[0]
                        new_q_value = NeuralNetwork.GAMMA * max(estimated_q_values)
                        y.append(new_q_value)
        # start the training
        self.model.fit(np.array(X), np.array(y), batch_size=NeuralNetwork.BATCH_SIZE,
                       epochs=NeuralNetwork.NB_EPOCHS, shuffle=True)
        # save the weights of the model
        if save_weights:
            self.model.save_weights('checkpoints/checkpoint_{}.h5py'.format(nb_checkpoint))

    @staticmethod
    def to_vector(state, action):
        """encodes a game state and an action as a vector

        :param state: the game state
        :param action: an action

        :type state: list[int]
        :type action: (int, int)

        :returns: an encoded vector
        :rtype: ndarray
        """
        x = np.zeros(18)
        x[:9] = state
        nb_action = action[0] * 3 + action[1]

        x[nb_action + 9] = 1  # One-hot encoding
        return x

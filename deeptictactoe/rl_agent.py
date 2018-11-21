from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from random import random, choice

from deeptictactoe.game import Player


class QAgent(Player):
    """This class implements a reinforcement learning algorithm."""

    def __init__(self, approximator, exploration_rate=0.3):
        """initializes the agent with some relevant parameters

        :param approximator: the approximator for the quality function
        :param exploration_rate: the exploration rate to use as a trade-off between exploration and exploitation
        :type exploration_rate: float
        :type approximator: QApproximator
        """
        self.approximator = approximator
        assert 0 <= exploration_rate <= 1, "The exploration rate has to be between 0 and 1."
        self.exploration_rate = exploration_rate

    def move(self, game_state, actions):
        if random() <= self.exploration_rate:
            return choice(actions)  # explore by choosing a random action from the action space
        # select the action with the highest Q value
        return max(actions, key=lambda a: self.approximator.Q(game_state, a))

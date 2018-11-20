from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from random import choice
from time import sleep

from six.moves import range


class Game:
    """This class implements the game of Tic-Tac-Toe for two players."""

    def __init__(self, player1, player2):
        """initializes the game with a cleared game state

        :param player1: the first player
        :param player2: the second player
        :type player1: Player
        :type player2: Player
        """
        self.players = (player1, player2)

        self.board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.current_player = 1
        self.history = {'1': [], '2': []}  # saves all moves and game states for each player
        self.result = -1  # 0 for a draw, 1 for a win of player 1, 2 for a win of player 2, -1 if still ongoing
        self.is_over = False

    def play(self, print_board=False, delay=0):
        """starts the game

        :param print_board: whether to print the board after each move or not
        :param delay: a delay in seconds added after each move

        :returns: the history of the game, i.e. all actions and the game´s result
        :rtype: dict
        """
        while not self.is_over:
            game_state = self.encode_game_state()
            actions = self.get_available_actions()
            row, column = self.players[self.current_player - 1].move(game_state, actions)

            # Check whether the move is legit
            if self.board[row][column] != 0:
                raise Exception(
                    "Player {} tried to mark field ({}|{]), but it was already marked.".format(self.current_player,
                                                                                               row + 1, column + 1))
            self.history[str(self.current_player)].append((game_state, actions, (row, column)))
            self.board[row][column] = self.current_player  # mark the field
            self.result = self.check_for_end()
            if print_board:
                self.print_board()
            if delay > 0:
                sleep(delay)  # a little delay is added after each move
            if self.result != -1:
                self.history['result'] = self.result
                self.is_over = True
                break
            self.current_player = 1 if self.current_player == 2 else 2  # The next player has to move.
        return self.history

    def check_for_end(self):
        """evaluates the current position and returns whether the game has ended and if so what was the result.
        -1 - The game is still going on.
         0 - The game ended in a draw.
         1 - Player 1 won the game.
         2 - Player 2 won the game.

        :return: int
        """
        # WIN CONDITION CHECKS
        # Check for three in a row.
        for row in self.board:
            if row[0] == row[1] == row[2] != 0:
                return row[0]

        # Check for three in a column
        for c in range(3):
            if self.board[0][c] == self.board[1][c] == self.board[2][c] != 0:
                return self.board[0][c]

        # Check the diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != 0:
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != 0:
            return self.board[0][2]

        # Check whether there are empty fields remaining on the board.
        for row in self.board:
            if 0 in row:
                return -1

        return 0  # There are no empty fields left.

    def encode_game_state(self):
        """encodes the current game state as a list
        The list contains nine integers, each representing one field of the board.
        0 indicates no one has marked the field yet
        1 indicates the current player to move has marked the field
        -1 indicated the opponent of the current player has marked the field

        :return: list[int]
        """
        return [1 if field == self.current_player else 0 if field == 0 else -1 for row in self.board for field in row]

    def get_available_actions(self):
        """returns all available actions for the current player to move
        These are simply all free fields on the board.

        :return: list[(int, int)]
        """
        actions = []
        for row in range(3):
            for column in range(3):
                if self.board[row][column] == 0:
                    actions.append((row, column))
        return actions

    def print_board(self):
        """prints the board to the terminal"""
        print("\n")
        for row in self.board:
            print("-------")
            print("|{}|{}|{}|".format(to_symbol(row[0]), to_symbol(row[1]), to_symbol(row[2])))
        print("-------")


class Player:
    """This class defines the functions the concrete implementations of agents have to implement."""

    def move(self, game_state, actions):
        """This function defines the behaviour of the agent as it returns its actions in a specific game state.

        :param game_state: the state of the game the agent should choose an action for
        :param actions: the available actions in the current game state, simply a list of all empty fields
        :type game_state: list[int]
        :type actions: list[(int, int)]

        :returns: the field the agent selects, specified by the numbers of its row and column
        :rtype: (int, int)
        """
        raise NotImplementedError("This method has to be implemented by the agents.")


class Randy(Player):
    """"This class implements a simple agent that selects a random action at each turn."""

    def move(self, game_state, actions):
        return choice(actions)


def to_symbol(number):
    """converts the number of a player to the corresponding symbol: 1 -> X, 2 -> O, 0 -> .

     :param number: the number of the player
     :type number: int

     :returns: the corresponding symbol for the specified number: X, O or .
     :return: str
     """
    return 'X' if number == 1 else 'O' if number == 2 else '.'

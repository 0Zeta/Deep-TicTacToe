from deeptictactoe.game import Game, Randy

game = Game(Randy(), Randy())
result = game.play(print_board=True, delay=1)
print(str(result))

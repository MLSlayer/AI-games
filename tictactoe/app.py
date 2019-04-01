from tictactoe.Game import Game

game = Game()

game.make_move((0, 0))
game.make_move((0, 1))
game.make_move((1, 1))
game.make_move((2, 1))
game.make_move((1, 2))
game.make_move((2, 2))
game.make_move((0, 2))
print(game)
game.make_move((2, 0))
print(game)
print(game.is_game_over())


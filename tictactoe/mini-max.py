from tictactoe.Game import *
import random

x_first = random.random() < .5
game = Game()

if (x_first):
    print("valid moves are ", game.valid_moves)
    move = int(input("player one(X), make first move: "))
    game.make_move(move)

player_dict = {1 : "one", 0: "two"}

def minimax(game):
    return next(iter(game.valid_moves.keys()))


while not game.is_game_over():
    print("valid moves are ", game.valid_moves)
    if game.player == X:
        move = int(input("player one(X), make move: "))
    else:
        move = minimax(game)
        print("player ", player_dict[game.player], " is now making move ", move)
    game.make_move(move)
    print(game)

print("player ", player_dict[game.get_winner()], " wins!")




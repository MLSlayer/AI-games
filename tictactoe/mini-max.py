from tictactoe.Game import *
import random

x_first = random.random() < .5
game = Game(X if x_first else O)

def minimax(game, i):
    if game.is_over() and not game.winner_exists():
        return 0
    if game.winner_exists():
        return 1 if game.get_winner() == O else -1
    valid_moves = game.valid_moves.copy().keys()
    best_move = next(iter(valid_moves))
    max_val = -10
    min_val = 10
    # keep history around for reminder(studying) purposes later :)
    history = []
    if i % 2 == 0:
        for m in valid_moves:
            game.make_move(m)
            next_move_val = minimax(game, i + 1)
            history.append((m, next_move_val))
            game.undo_move(m)
            if next_move_val > max_val:
                max_val = next_move_val
                best_move = m
    else:
        for m in valid_moves:
            game.make_move(m)
            next_move_val = minimax(game, i + 1)
            history.append((m, next_move_val))
            game.undo_move(m)
            if next_move_val < min_val:
                min_val = next_move_val
                best_move = m
    if i == 0:
        return best_move

    return max_val if i % 2 == 0 else min_val

if (x_first):
    print("valid moves are ", game.valid_moves)
    move = int(input("player one(X), make first move: "))
    game.make_move(move)

while not game.is_over():
    if game.player == X:
        print("valid moves are ", game.valid_moves)
        move = int(input("please make move: "))
    else:
        move = minimax(game, 0)
        print("computer is now making move ", move)
    game.make_move(move)
    print(game)

if game.winner_exists():
    print("You win!" if game.get_winner() == X else "You lose!")
else:
    print("player one and two ties")




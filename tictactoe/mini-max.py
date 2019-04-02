from tictactoe.Game import *
import random

player_first = random.random() < .5
game = Game(X if player_first else O)

player_dict = {X : "one" if player_first else "two",
               O : "two" if player_first else "one"}

def minimax(game, max_player):
    def aux(game, move, depth):
        if move is not None:
            game.make_move(move)

        valid_moves = game.valid_moves.copy().keys()
        if not valid_moves:
            game.undo_move(move)
            if depth == 1:
                return move, (move, 0)
            return move, 0, True

        if game.winner_exists():
            winner = game.get_winner()
            game.undo_move(move)
            if depth == 1:
                return move, (move, 1)
            return (move, 1, True) if winner == max_player else (move, -1, True)

        moves = [aux(game, m, depth + 1) for m in valid_moves]
        if depth is not 1 and depth is not 0:
            term_moves = list(map(lambda m : m[1], filter(lambda m: m[2], moves)))
            moves = list(map(lambda m: m[1], filter(lambda m: not m[2], moves)))
            moves.append(sum(term_moves))
            game.undo_move(move)
            return (move, min(moves), False) if game.player == X else (move, max(moves), False)

        if depth == 0:
            return max(moves, key = lambda m : m[1][1])

        term_moves = list(map(lambda m : (m[0],m[1]), filter(lambda m: m[2], moves)))
        moves = list(map(lambda m: (m[0],m[1]), filter(lambda m: not m[2], moves)))
        moves.extend(term_moves)
        game.undo_move(move)
        return move, min(moves, key = lambda m : m[1])

    move = aux(game, None, 0)[0]
    return move

if (player_first):
    print("valid moves are ", game.valid_moves)
    move = int(input("player one(X), make first move: "))
    game.make_move(move)
else:
    move = minimax(game, O if player_first else X)
    print("player ", player_dict[game.player], " is now making move ", move)


while not game.is_over():
    if game.player == X:
        print("valid moves are ", game.valid_moves)
        move = int(input("player one(X), make move: "))
    else:
        move = minimax(game, O if player_first else X)
        print("player ", player_dict[game.player], " is now making move ", move)
    game.make_move(move)
    print(game)

if game.winner_exists():
    print("player ", player_dict[game.get_winner()], " wins!")
else:
    print("player one and two ties")




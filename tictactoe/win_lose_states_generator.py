from tictactoe.Game import *
import pandas as pd

board_to_str = lambda b: ''.join(map(lambda s: str(s), b))

# for when O goes first
def minimax(game, i, move_confs):
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
            next_move_val = minimax(game, i + 1, move_confs)
            history.append((m, next_move_val))
            game.undo_move(m)
            if next_move_val > max_val:
                max_val = next_move_val
                best_move = m
    else:
        for m in valid_moves:
            game.make_move(m)
            next_move_val = minimax(game, i + 1, move_confs)
            history.append((m, next_move_val))
            game.undo_move(m)
            if next_move_val < min_val:
                min_val = next_move_val
                best_move = m
    if i % 2 == 0:
        move_confs.add((board_to_str(game.board.reshape(-1)), max_val))
    else:
        move_confs.add((board_to_str(game.board.reshape(-1)), min_val))
    if i == 0:
        return best_move

    return max_val if i % 2 == 0 else min_val

game = Game(O)
move_confs = set()
minimax(game, 0, move_confs)

move_confs = pd.DataFrame(list(move_confs), columns=["board_state", "score"])
move_confs.to_csv("q_learning_meta/pc_first_win_lose_states.txt", index=False)


# for when X goes first
game = Game(X)
move_confs = set()
valid_moves = game.valid_moves.copy().keys()
for move in valid_moves:
    game.make_move(move)
    minimax(game, 0, move_confs)
    game.undo_move(move)

move_confs = pd.DataFrame(list(move_confs), columns=["board_state", "score"])
move_confs.to_csv("q_learning_meta/human_first_win_lose_states.txt", index=False)


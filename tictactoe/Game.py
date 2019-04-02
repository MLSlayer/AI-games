import numpy as np
import re
from tictactoe.Marks import *


class Game(object):
    def __init__(self, first_player):
        self.board = -np.ones((3, 3), dtype=int)
        self.player = first_player
        self.moves = dict(enumerate([(i,j) for i in range(3) for j in range(3)]))
        self.valid_moves = self.moves.copy()
        self.is_tie = False

    def make_move(self, move):
        assert move in self.valid_moves
        assert not self.winner_exists()

        i, j = self.moves[move]
        self.board[i][j] = self.player
        self.player = O if self.player == X else X
        self.valid_moves.pop(move)

    def undo_move(self, move):
        assert move not in self.valid_moves
        self.valid_moves[move] = self.moves[move]
        i, j = self.moves[move]
        self.board[i][j] = EMPTY
        self.player = O if self.player == X else X

    def get_winner(self):
        assert self.winner_exists()
        return O if self.player == X else X

    def is_over(self):
        return not self.valid_moves or self.winner_exists()

    def winner_exists(self):
        is_over = any([len(set(row)) <= 1 and EMPTY not in set(row) for row in self.board])
        is_over = any([len(set(col)) <= 1 and EMPTY not in set(col) for col in (self.board.T)]) or is_over
        is_over = len(set(np.diag(self.board))) <= 1 and EMPTY not in set(np.diag(self.board)) or is_over
        is_over = len(set(np.diag(np.fliplr(self.board)))) <= 1 and EMPTY not in set(np.diag(np.fliplr(self.board))) or is_over
        return is_over

    def __repr__(self):
        s = re.sub(str(EMPTY), "  ", str(self.board))
        s = re.sub(str(X), "X", s)
        s = re.sub(str(O), "O", s)
        return  s + "\n"

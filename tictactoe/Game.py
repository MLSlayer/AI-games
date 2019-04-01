import numpy as np
import re
from tictactoe.Marks import *


class Game(object):
    def __init__(self):
        self.board = -np.ones((3, 3), dtype=int)
        self.player = X
        self.winner = None

    def make_move(self, pos):
        i, j = pos
        assert self.board[i][j] == EMPTY
        assert not self.is_game_over()

        self.board[i][j] = self.player
        if self.is_game_over():
            self.winner = self.player
        self.player = O if self.player == X else X

    def is_game_over(self):
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

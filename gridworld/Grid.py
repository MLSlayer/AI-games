import random
import numpy as np
import re

class Grid:
    def __init__(self, static=False):
        if static:
            self.win_pos = (0, 3)
            self.lose_pos = (1,3)
            self.wall_pos = (1, 1)
            self.cur_pos = (2, 0)
        else:
            while True:
                w, l, wl, c = random.sample([(i,j) for i in range(3) for j in range(4)], 4)
                self.win_pos = w
                self.lose_pos = l
                self.wall_pos = wl
                self.cur_pos = c
                if self.can_win():
                    break
        self.is_over = False

    def get_score(self):
        if self.cur_pos == self.win_pos:
            self.is_over = True
            return 1
        if self.cur_pos == self.lose_pos:
            self.is_over = True
            return -1
        return 0

    def move_up(self):
        pos = self.cur_pos[0] - 1, self.cur_pos[1]
        return self.move_aux(pos)

    def move_down(self):
        pos = self.cur_pos[0] + 1, self.cur_pos[1]
        return self.move_aux(pos)

    def move_left(self):
        pos = self.cur_pos[0], self.cur_pos[1] - 1
        return self.move_aux(pos)

    def move_right(self):
        pos = self.cur_pos[0], self.cur_pos[1] + 1
        return self.move_aux(pos)

    def move_aux(self, pos):
        assert not self.is_over
        if not self.pos_is_valid((pos)):
            return 0
        self.cur_pos = pos
        return self.get_score()

    def pos_is_valid(self, pos):
        i,j = pos
        return i >= 0 and i <= 2 and j >= 0 and j <= 4 and self.wall_pos != pos

    def can_win(self):
        if (self.win_pos == (2, 0) or self.cur_pos == (2, 0)) and (self.wall_pos == (1, 0) and self.lose_pos == (2, 1) or self.wall_pos == (2, 1) and self.lose_pos == (1, 0)):
            return False
        if (self.win_pos == (0, 3) or self.cur_pos == (0, 3)) and (self.wall_pos == (0, 2) and self.lose_pos == (1, 3) or self.wall_pos == (3, 1) and self.lose_pos == (0, 2)):
            return False
        if (self.win_pos == (0, 0) or self.cur_pos == (0, 0)) and (self.wall_pos == (1, 0) and self.lose_pos == (0, 1) or self.wall_pos == (0, 1) and self.lose_pos == (1, 0)):
            return False
        if (self.win_pos == (2, 3) or self.cur_pos == (2, 3)) and (self.wall_pos == (2, 2) and self.lose_pos == (1, 3) or self.wall_pos == (1, 3) and self.lose_pos == (2, 2)):
            return False
        return True

    def print_grid(self):
        grid = np.chararray((3,4))
        grid[:] = "0"
        grid[self.win_pos] = "1"
        grid[self.cur_pos] = "p"
        grid[self.lose_pos] = "-1"
        grid[self.wall_pos] = "w"
        print(re.sub(r'[b\']', "", str(grid)), "\n")







































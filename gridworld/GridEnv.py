from gym.envs.toy_text import discrete
import numpy as np

UP, RIGHT, DOWN, LEFT = range(4)

class GridEnv(discrete.DiscreteEnv):

    def __init__(self, nS, nA, P, isd):
        super().__init__(nS, nA, P, isd)

    @classmethod
    def static(cls):
        # static builds the below, p is the person, w is the wall, 1 and -1 are the terminal states
        #       0  0  0  1                                                u d l r
        #       0  w  0 -1     actions/policies in client program ->      u x l r       where u,d,l,r,x = up,down,left,right,skip-manually-in-code
        #       0  0  0  0                                                u d l r       also, w gives reward zero, but I just marked it w to show you its a wall
        nS = 12
        P = {}

        for s in range(nS):
            P[s] = {}
            is_wall = lambda p: p == 5
            get_reward = lambda p: 1 if p == 3 else -1 if p == 7 else 0

            top = s if is_wall(s - 4) or s in np.arange(4) else s - 4
            right = s if is_wall(s + 1) or s % 4 == 3 else s + 1
            bottom = s if is_wall(s + 4) or s in np.arange(8, 16) else s + 4
            left = s if is_wall(s - 1) or s % 4 == 0 else s - 1

            # breaking the general P pattern, leaving out probabilty and also making the value of P[x][y] a tuple instead of a list
            P[s][0] = (top, get_reward(top), get_reward(top) == 1)
            P[s][1] = (right, get_reward(right), get_reward(right) == 1)
            P[s][2] = (bottom, get_reward(bottom), get_reward(bottom) == 1)
            P[s][3] = (left, get_reward(left), get_reward(left) == 1)

            if s == 3 or s == 7:
                P[s][0] = (s, 0, True)
                P[s][1] = (s, 0, True)
                P[s][2] = (s, 0, True)
                P[s][3] = (s, 0, True)

        return cls(nS, 4, P, np.ones(nS) / nS)





from gym.envs.toy_text import discrete
import numpy as np

NA = 4
UP, RIGHT, DOWN, LEFT = range(NA)

class GridEnv(discrete.DiscreteEnv):

    def __init__(self, nS, nA, P, isd):
        super().__init__(nS, nA, P, isd)

    @classmethod
    def static(cls):
        # static builds the below, p is the person, w is the wall, 1 and -1 are the terminal states
        #       0  0  0  1                                                u d l r
        #       0  w  0 -1     actions/policies in client program ->      u d l r       where u,d,l,r = up,down,left,right
        #       0  0  0  0                                                u d l r       also, w gives reward zero, but I just marked it w to show you its a wall
        nS = 12
        P = {}
        nA = NA

        for s in range(nS):
            P[s] = {}
            is_wall = lambda p: p == 5
            get_reward = lambda p: 1 if p == 3 else -1 if p == 7 else 0

            top = s if is_wall(s - nA) or s in np.arange(nA) else s - nA
            right = s if is_wall(s + 1) or s % nA == 3 else s + 1
            bottom = s if is_wall(s + nA) or s in np.arange(8, 12) else s + nA
            left = s if is_wall(s - 1) or s % nA == 0 else s - 1

            # breaking the general P pattern, leaving out probabilty and also making the value of P[x][y] a tuple instead of a list
            # this will break things such as calling step, but it's just to peel one more layer off when first trying to understand policy_iteration and value_iteration
            # steppable_static is a more proper implementation
            P[s][0] = (top, get_reward(top), get_reward(top) == 1)
            P[s][1] = (right, get_reward(right), get_reward(right) == 1)
            P[s][2] = (bottom, get_reward(bottom), get_reward(bottom) == 1)
            P[s][3] = (left, get_reward(left), get_reward(left) == 1)

            if s == 3 or s == 7 or s == 5:
                P[s][0] = (s, 0, True)
                P[s][1] = (s, 0, True)
                P[s][2] = (s, 0, True)
                P[s][3] = (s, 0, True)

        return cls(nS, nA, P, np.ones(nS) / nS)

    @classmethod
    def steppable_static(cls):
        nS = 12
        P = {}
        nA = NA

        for s in range(nS):
            P[s] = {}
            is_wall = lambda p: p == 5
            get_reward = lambda p: 1 if p == 3 else -1 if p == 7 else 0

            top = lambda s: s if is_wall(s - nA) or s in np.arange(nA) else s - nA
            right = lambda s: s if is_wall(s + 1) or s % nA == 3 else s + 1
            bottom = lambda s: s if is_wall(s + nA) or s in np.arange(8, 12) else s + nA
            left = lambda s: s if is_wall(s - 1) or s % nA == 0 else s - 1

            for a in range(nA):
                moves = []
                r, b, l, t = right(s), bottom(s), left(s), top(s)
                is_done = lambda s: s == 3 or s == 5 or s == 7
                get_move_s = lambda s : (1/3, s, get_reward(s), is_done(s))
                if a == UP:
                    moves.append((get_move_s(l)))
                    moves.append((get_move_s(r)))
                    moves.append((get_move_s(t)))
                if a == RIGHT:
                    moves.append((get_move_s(t)))
                    moves.append((get_move_s(b)))
                    moves.append((get_move_s(r)))
                if a == DOWN:
                    moves.append((get_move_s(l)))
                    moves.append((get_move_s(r)))
                    moves.append((get_move_s(b)))
                if a == LEFT:
                    moves.append((get_move_s(t)))
                    moves.append((get_move_s(b)))
                    moves.append((get_move_s(l)))
                P[s][a] = moves

        for s in [3, 5, 7]:
            for a in range(4):
                P[s][a] = [(1, s, 0, True)]

        return cls(nS, nA, P, np.ones(nS) / nS)


def get_prob_of_success(env, pi, n):
    win = 0
    for i in range(n):
        s, is_done = env.reset(), False
        while not is_done:
            s, reward, is_done, _ = env.step(pi[s])
            if reward == 1:
                win += 1
    return win / n


































from gridworld.GridEnv import *
import numpy as np


def gen_episode(pi, env, max_steps):
    episode = []
    s = env.reset()

    for t in range(max_steps):
        a = pi[s]
        new_s, reward, is_done, _ = env.step(a)
        episode.append((s, a, reward, new_s, is_done))
        if is_done:
            break
        s = new_s
    return np.array(episode)


game = GridEnv.steppable_static()
LEFT, DOWN, RIGHT, UP = range(4)
pi = [0, 1, 2, 3,
      0, 1, 2, 3,
      0, 1, 2, 3]

episode = gen_episode(pi, game, 20)
print(episode)

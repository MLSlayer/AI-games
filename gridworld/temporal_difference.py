from gridworld.GridEnv import *

def get_state_values_td(pi, env, gamma=0.9, alpha=0.2, alpha_decay_rate=.0003, min_alpha=0, episodes=30000):
    nS = env.nS
    V = np.zeros(nS)

    for t in range(episodes):
        alpha = max(min_alpha, alpha * np.exp(-alpha_decay_rate * t))

        s = env.reset()
        is_done = False

        while not is_done:
            a = pi[s]
            new_s, reward, is_done, _ = env.step(a)

            td_error = reward + gamma * V[new_s] - V[s]
            V[s] += alpha * td_error
            s = new_s
    return V

game = GridEnv.steppable_static()
LEFT, DOWN, RIGHT, UP = range(4)
pi = [0, 1, 2, 3,
      0, 1, 2, 3,
      0, 1, 2, 3]


V = get_state_values_td(pi, game, alpha=1)
print(V)

# less variance, and close to the true state values, the bias isn't that bad either...
# [ 0.10325534  0.12150854  0.16389367  0.          0.08338727  0.
#   -0.65685648  0.         -0.10264621 -0.30644452 -0.41139743 -0.65096012]

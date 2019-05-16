from gridworld.GridEnv import *


def gen_episode(pi, env, max_steps=500):
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


def get_state_values_mc(pi, env, gamma=0.9, alpha=0.2, min_alpha=0, alpha_decay_rate=0.0003, episodes=30000, max_steps=500):
    nS = env.nS
    discounts = np.logspace(0, max_steps, num=max_steps, base=gamma, endpoint=False)
    V = np.zeros(nS)

    for t in range(episodes):
        alpha = max(min_alpha, alpha * np.exp(-alpha_decay_rate * t))
                                    # >>> np.exp(-.0003 * 10000) = 0.049787068367863965
                                    # >>> np.exp(-.0003 * 20000) = 0.0024787521766663607
        episode = gen_episode(pi, env, max_steps)
        visited = set()

        for i, (s, a, reward, new_s, is_done) in enumerate(episode):
            if s in visited:
                continue
            visited.add(s)

            G = np.sum(discounts[:len(episode) - i] * episode[i:, 2])
            V[s] = V[s] + alpha * (G - V[s])
    return V

game = GridEnv.steppable_static()
LEFT, DOWN, RIGHT, UP = range(4)
pi = [0, 1, 2, 3,
      0, 1, 2, 3,
      0, 1, 2, 3]

# episode = gen_episode(pi, game, 20)
# print(episode)

V = get_state_values_mc(pi, game)
print(V)

# [ 0.08714857  0.10060156  0.08552832  0.          0.02409157  0.
#   -0.58712062  0.         -0.09870475 -0.33590452 -0.37950307 -0.63593771]


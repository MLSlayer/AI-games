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
            V[s] += alpha * (G - V[s])
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


# Epsilon greedy action selection over Q(s,a), for more detail on soft policies checkout below
# https://stats.stackexchange.com/questions/342379/what-are-soft-policies-in-reinforcement-learning
class ESoftPolicy:
    def __init__(self, nS, nA, eps=0.2):
        assert nA > 1 and nS > 0
        self.nS = nS
        self.nA = nA
        self.probs = np.zeros((nS, nA))

        for s, _ in enumerate(self.probs):
            self.reset_prob(s, np.random.randint(nA), eps)


    def reset_prob(self, s, best_a, eps):
        nS, nA = self.nS, self.nA
        one_minus_eps = 1 - eps
        shared_eps = eps / (nA - 1)

        self.probs[s][:] = shared_eps
        self.probs[s][best_a] = one_minus_eps

    def __getitem__(self, s):
        return np.random.choice(self.nA, p=self.probs[s])


def on_policy_mc_control(env, gamma=0.9, eps=1, min_eps=0.005, eps_decay_rate=.00015,
                         alpha=.7, min_alpha=0.0005, alpha_decay_rate=.00015, episodes=70000, max_steps=500):
    nS, nA = env.nS, env.nA
    discounts = np.logspace(0, max_steps, base=gamma, num=max_steps, endpoint=False)

    Q = np.zeros((nS, nA))
    pi = ESoftPolicy(nS, nA, eps)

    for t in range(episodes):
        eps = max(min_eps, eps * np.exp(-eps_decay_rate * t))
        alpha = max(min_alpha, alpha * np.exp(-alpha_decay_rate * t))
        episode = gen_episode(pi, env, max_steps)

        visited = set()
        for i, (s, a, reward, _, _) in enumerate(episode):
            if (s,a) in visited:
                continue
            visited.add((s,a))

            G = np.sum(discounts[:len(episode) - i] * episode[i:, 2])
            Q[s][a] += alpha * (G - Q[s][a])

        _, first_indices = np.unique(episode[:, 0], return_index=True)
        for idx in first_indices:
            s, a, reward, _, _ = episode[idx]
            best_a = np.argmax(Q[s])
            pi.reset_prob(s, best_a, eps)

    V = np.max(Q, axis=1)
    return Q, V

Q, V = on_policy_mc_control(game)
pi = [a for a in np.argmax(Q, axis=1)]
print("-------------------------")
print("pi is ", pi)
print("-------------------------")
print("Q is ", Q)
print("-------------------------")
print("V is ", V)


# -------------------------
# pi is  [0, 2, 0, 0, 3, 0, 3, 0, 0, 1, 3, 2]
# -------------------------
# Q is  [[ 3.04849174e-01  2.36437394e-01  2.50922598e-01  1.61911894e-01]
#        [ 3.48977161e-01  3.96823081e-01  4.07417580e-01  3.17388539e-01]
# [ 6.60036174e-01  3.18744790e-01 -7.94097888e-03  3.71185745e-01]
# [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00]
# [ 1.21251258e-01 -4.57612607e-02  4.51235319e-02  1.94562982e-01]
# [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00]
# [-6.36379136e-02 -6.53651948e-01 -6.03093837e-01  3.85261029e-01]
# [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00]
# [ 1.57612444e-01 -1.19897981e-02 -7.11198768e-02 -8.49787863e-02]
# [ 8.44222648e-02  1.78832312e-01  1.07123751e-02 -2.07855680e-04]
# [-3.66677992e-01 -1.28046013e-01 -2.92113446e-01  2.38476999e-01]
# [-5.87462595e-01 -8.18465187e-01  1.55544858e-01 -7.65075893e-01]]
# -------------------------
# V is  [0.30484917 0.40741758 0.66003617 0.         0.19456298 0.
#        0.38526103 0.         0.15761244 0.17883231 0.238477   0.15554486]


get_prob_of_success(game, pi, 15000)

# 0.7596

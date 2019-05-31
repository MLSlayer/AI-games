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


def on_policy_mc_control(env, gamma=0.9, initial_eps=1, min_eps=0.005, eps_decay_rate=.00015,
                         initial_alpha=.7, min_alpha=0.0005, alpha_decay_rate=.00015, episodes=90000, max_steps=500):
    nS, nA = env.nS, env.nA
    discounts = np.logspace(0, max_steps, base=gamma, num=max_steps, endpoint=False)

    Q = np.zeros((nS, nA))
    pi = ESoftPolicy(nS, nA, initial_eps)

    for t in range(episodes):
        eps = max(min_eps, initial_eps * np.exp(-eps_decay_rate * t))
        alpha = max(min_alpha, initial_alpha * np.exp(-alpha_decay_rate * t))
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
print(get_prob_of_success(game, pi, 15000))

# [ 0.21729829  0.26834734  0.30562068  0.          0.17155367  0.
#   -0.54726455  0.         -0.03807383 -0.25033826 -0.41494966 -0.62983943]
# -------------------------
# pi is  [0, 1, 0, 0, 3, 0, 3, 0, 0, 0, 1, 2]
# -------------------------
# Q is  [[ 0.38753441  0.31601659  0.32551386  0.27739647]
#        [ 0.46978057  0.520514    0.46230936  0.40435669]
# [ 0.69273459  0.64564939  0.60652181  0.43672285]
# [ 0.          0.          0.          0.        ]
# [ 0.21502925  0.22526045  0.18560517  0.24347153]
# [ 0.          0.          0.          0.        ]
# [-0.03330975 -0.00763457 -0.0682704   0.400778  ]
# [ 0.          0.          0.          0.        ]
# [ 0.1850012   0.16140701  0.15362008  0.17130925]
# [ 0.18329231  0.14415393  0.13545408  0.13512504]
# [ 0.17980572  0.24663445  0.1668752   0.21166215]
# [-0.17791628 -0.26087857  0.18203998 -0.16685301]]
# -------------------------
# V is  [0.38753441 0.520514   0.69273459 0.         0.24347153 0.
#        0.400778   0.         0.1850012  0.18329231 0.24663445 0.18203998]

# 0.7478666666666667

# (V - np.min(V)) / (np.max(V) - np.min(V))
# array([0.55942697, 0.75139023, 1.        , 0.        , 0.35146438,
#        0.        , 0.57854481, 0.        , 0.26705928, 0.2645924 ,
#        0.35603022, 0.2627846 ])



from gridworld.GridEnv import *

def q_learning(env, gamma=.9, initial_alpha=.5, min_alpha = .0005, alpha_decay_rate=.002,
               initial_eps=1, eps_decay_rate=.00005, min_eps=.005, episodes=70000):
    nS, nA = env.nS, env.nA
    Q = np.zeros((nS, nA))

    def select_a_strategy(s, Q, eps):
        if np.random.rand() > eps:
            return np.argmax(Q[s])
        return np.random.randint(nA)

    for t in range(episodes):
        alpha = max(min_alpha, initial_alpha * np.exp(-alpha_decay_rate * t))
        eps = max(min_eps, initial_eps * np.exp(-eps_decay_rate * t))
        s, is_done = env.reset(), False

        while not is_done:
            a = select_a_strategy(s, Q, eps)
            new_s, reward, is_done, _ = env.step(a)

            if is_done:
                Q[new_s] = 0
                td_error = reward - Q[s][a]
            else:
                td_error = reward + gamma * Q[new_s].max() - Q[s][a]
            Q[s][a] += alpha * td_error
            s = new_s

    V = np.max(Q, axis=1)
    return Q, V

game = GridEnv.steppable_static()
LEFT, DOWN, RIGHT, UP = range(4)

Q, V = q_learning(game)
pi = [a for a in np.argmax(Q, axis=1)]
print("-------------------------")
print("pi is ", pi)
print("-------------------------")
print("Q is ", Q)
print("-------------------------")
print("V is ", V)


print(get_prob_of_success(game, pi, 15000))
# -------------------------
# pi is  [0, 1, 0, 0, 0, 0, 3, 0, 3, 1, 3, 2]
# -------------------------
# Q is  [[ 0.39600897  0.36697971  0.36680082  0.3277363 ]
#        [ 0.48702271  0.52937488  0.48741917  0.43608169]
# [ 0.70529647  0.67774974  0.61857715  0.49669997]
# [ 0.          0.          0.          0.        ]
# [ 0.2969622   0.27807945  0.24649726  0.2745978 ]
# [ 0.          0.          0.          0.        ]
# [-0.00313581 -0.05221431 -0.11668463  0.41487024]
# [ 0.          0.          0.          0.        ]
# [ 0.22073884  0.22075808  0.20152217  0.22196264]
# [ 0.20975071  0.20975179  0.20975054  0.19995577]
# [ 0.25393803  0.26556429  0.21133676  0.26561477]
# [-0.15314242 -0.18799467  0.20091118 -0.17877197]]
# -------------------------
# V is  [0.39600897 0.52937488 0.70529647 0.         0.2969622  0.
#        0.41487024 0.         0.22196264 0.20975179 0.26561477 0.20091118]
# 0.7500666666666667

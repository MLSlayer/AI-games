from gridworld.GridEnv import *

def double_q_learning(env, gamma=.9, initial_alpha=.5, min_alpha = .0005, alpha_decay_rate=.002,
               initial_eps=1, eps_decay_rate=.00005, min_eps=.005, episodes=70000):
    nS, nA = env.nS, env.nA
    Q1 = np.zeros((nS, nA))
    Q2 = np.zeros((nS, nA))

    def select_a_strategy(s, Q, eps):
        if np.random.rand() > eps:
            return np.argmax(Q[s])
        return np.random.randint(nA)

    for t in range(episodes):
        alpha = max(min_alpha, initial_alpha * np.exp(-alpha_decay_rate * t))
        eps = max(min_eps, initial_eps * np.exp(-eps_decay_rate * t))
        s, is_done = env.reset(), False

        while not is_done:
            a = select_a_strategy(s, Q1 + Q2, eps)
            new_s, reward, is_done, _ = env.step(a)

            if is_done:
                Q1[new_s] = 0
                Q2[new_s] = 0
            if np.random.randint(2):
                if is_done:
                    td_error = reward - Q1[s][a]
                else:
                    q1_argmax = np.argmax(Q1[new_s])
                    td_error = reward + gamma * Q2[new_s][q1_argmax] - Q1[s][a]
                Q1[s][a] += alpha * td_error
            else:
                if is_done:
                    td_error = reward - Q2[s][a]
                else:
                    q2_argmax = np.argmax(Q2[new_s])
                    td_error = reward + gamma * Q1[new_s][q2_argmax] - Q2[s][a]
                Q2[s][a] += alpha * td_error
            s = new_s

    Q = (Q1 + Q2) / 2
    V = np.max(Q, axis=1)
    return Q, V

game = GridEnv.steppable_static()
LEFT, DOWN, RIGHT, UP = range(4)

Q, V = double_q_learning(game)
pi = [a for a in np.argmax(Q, axis=1)]
print("-------------------------")
print("pi is ", pi)
print("-------------------------")
print("Q is ", Q)
print("-------------------------")
print("V is ", V)


print(get_prob_of_success(game, pi, 15000))

# -------------------------
# pi is  [0, 1, 0, 0, 0, 0, 3, 0, 3, 0, 3, 2]
# -------------------------
# Q is  [[ 0.39304447  0.35087028  0.34967207  0.31005729]
#        [ 0.48092997  0.52706205  0.4795188   0.42714108]
# [ 0.69805564  0.65660987  0.60825188  0.47564394]
# [ 0.          0.          0.          0.        ]
# [ 0.29419505  0.25660906  0.22772265  0.25759571]
# [ 0.          0.          0.          0.        ]
# [-0.01969484 -0.08463284 -0.14957791  0.41362152]
# [ 0.          0.          0.          0.        ]
# [ 0.2013178   0.20178779  0.17892802  0.21927985]
# [ 0.20552408  0.18953624  0.19225146  0.17525536]
# [ 0.22916779  0.2479928   0.18838083  0.26518688]
# [-0.20338622 -0.23386795  0.19272166 -0.21623326]]
# -------------------------
# V is  [0.39304447 0.52706205 0.69805564 0.         0.29419505 0.
#        0.41362152 0.         0.21927985 0.20552408 0.26518688 0.19272166]
# 0.7516


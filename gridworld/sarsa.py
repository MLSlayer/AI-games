from gridworld.GridEnv import *

def sarsa(env, gamma=0.9, alpha=1, min_alpha=.0005, alpha_decay_rate=.0001,
          eps=1, eps_decay_rate=.0001, min_eps=.005, episodes=70000):
    nS, nA = env.nS, env.nA
    Q = np.zeros((nS, nA))

    def select_a_strategy(s, Q, eps):
        if np.random.rand() > eps:
            return np.argmax(Q[s])
        return np.random.randint(nA)

    for t in range(episodes):
        alpha = max(min_alpha, alpha * np.exp(-alpha_decay_rate * t))
        eps = max(min_eps, eps * np.exp(-eps_decay_rate * t))
        s, is_done = env.reset(), False
        a = select_a_strategy(s, Q, eps)

        while not is_done:
            new_s, reward, is_done, _ = env.step(a)
            new_a = select_a_strategy(new_s, Q, eps)

            if is_done:
                Q[new_s] = 0
                td_error = reward - Q[s][a]
            else:
                td_error = reward + gamma * Q[new_s][new_a] - Q[s][a]
            Q[s][a] += alpha * td_error
            s, a = new_s, new_a

    V = np.max(Q, axis=1)
    return Q, V


game = GridEnv.steppable_static()
LEFT, DOWN, RIGHT, UP = range(4)

Q, V = sarsa(game)
pi = [a for a in np.argmax(Q, axis=1)]
print("-------------------------")
print("pi is ", pi)
print("-------------------------")
print("Q is ", Q)
print("-------------------------")
print("V is ", V)


print(get_prob_of_success(game, pi, 15000))

# -------------------------
# pi is  [0, 1, 0, 0, 0, 0, 3, 0, 3, 3, 3, 2]
# -------------------------
# Q is  [[ 0.39086433  0.15185376  0.23017024  0.14186986]
#        [ 0.31685682  0.51990032  0.3928658   0.31823102]
# [ 0.69745836  0.47142299  0.41340415  0.18162964]
# [ 0.          0.          0.          0.        ]
# [ 0.29085144  0.09291243  0.03291805  0.09186377]
# [ 0.          0.          0.          0.        ]
# [-0.01844164 -0.50356041 -0.70971098  0.40028481]
# [ 0.          0.          0.          0.        ]
# [-0.02542957 -0.01831781 -0.01993207  0.21807647]
# [-0.06868493 -0.13206473 -0.05537995  0.16107552]
# [-0.35714582 -0.28563931 -0.28467809  0.23597476]
# [-0.83368377 -0.53889402  0.14761193 -0.72017302]]
# -------------------------
# V is  [0.39086433 0.51990032 0.69745836 0.         0.29085144 0.
#        0.40028481 0.         0.21807647 0.16107552 0.23597476 0.14761193]
# 0.7488



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
        alpha = min(min_alpha, alpha * np.exp(-alpha_decay_rate * t))
        eps = min(min_eps, eps * np.exp(-eps_decay_rate * t))
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
# pi is  [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 2]
# -------------------------
# Q is  [[ 3.46963545e-05  0.00000000e+00  0.00000000e+00  2.02386858e-09]
#        [ 1.24313759e-03  1.43767414e-05  2.09211020e-06  2.92392854e-07]
# [ 4.05073437e-02  2.40211411e-06  0.00000000e+00  7.46630886e-09]
# [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00]
# [ 1.42242360e-07  0.00000000e+00  0.00000000e+00  0.00000000e+00]
# [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00]
# [-4.89692345e-04 -4.75692559e-04 -4.71117767e-04  2.26982001e-04]
# [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00]
# [ 3.27952899e-10  0.00000000e+00  3.22687880e-16  0.00000000e+00]
# [ 4.66834852e-09  0.00000000e+00  0.00000000e+00  0.00000000e+00]
# [ 1.33214278e-06  0.00000000e+00  0.00000000e+00  0.00000000e+00]
# [-4.77282280e-04 -4.72722293e-04  7.55671346e-09  0.00000000e+00]]
# -------------------------
# V is  [3.46963545e-05 1.24313759e-03 4.05073437e-02 0.00000000e+00
#        1.42242360e-07 0.00000000e+00 2.26982001e-04 0.00000000e+00
#        3.27952899e-10 4.66834852e-09 1.33214278e-06 7.55671346e-09]
# 0.7528

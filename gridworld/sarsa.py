from gridworld.GridEnv import *

def sarsa(env, gamma=0.9, initial_alpha=1, min_alpha=.0005, alpha_decay_rate=.0001,
          initial_eps=1, eps_decay_rate=.0001, min_eps=.005, episodes=90000):
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
# pi is  [0, 1, 0, 0, 0, 0, 3, 0, 3, 2, 3, 2]
# -------------------------
# Q is  [[ 0.39415454  0.35137826  0.36051143  0.31189086]
#        [ 0.4768882   0.5264558   0.48580826  0.4287576 ]
# [ 0.70510662  0.65867683  0.60548407  0.47638796]
# [ 0.          0.          0.          0.        ]
# [ 0.29255546  0.25192543  0.22680108  0.26655026]
# [ 0.          0.          0.          0.        ]
# [-0.0691952  -0.05764888 -0.06376372  0.41554123]
# [ 0.          0.          0.          0.        ]
# [ 0.20506866  0.19937805  0.18372799  0.22031379]
# [ 0.18790235  0.18563357  0.2110992   0.17940156]
# [ 0.22582434  0.22999671  0.18813503  0.27134116]
# [-0.24950507 -0.3729818   0.19736997 -0.28590823]]
# -------------------------
# V is  [0.39415454 0.5264558  0.70510662 0.         0.29255546 0.
#        0.41554123 0.         0.22031379 0.2110992  0.27134116 0.19736997]
# 0.7539333333333333



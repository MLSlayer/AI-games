from gridworld.GridEnv import *


def get_state_values(pi, P, gamma=0.9, theta=1e-10):
    V = np.zeros(len(pi))

    while True:
        delta = 0
        aux_V = np.zeros(len(pi))

        for s in range(len(P)):
            a = pi[s]
            for prob, new_s, reward, is_done in P[s][a]:
                if is_done:
                    value = reward
                else:
                    value = reward + gamma * V[new_s]

                aux_V[s] += prob * value
            delta = max(delta, abs(aux_V[s] - V[s]))

        V = aux_V

        if delta < theta:
            return V

game = GridEnv.steppable_static()
pi = [0, 1, 2, 3,
      0, 1, 2, 3,
      0, 1, 2, 3]

V = get_state_values(pi, game.P)
print(V)

# [ 0.09976721  0.13302295  0.17736393  0.          0.07482541  0.
#   -0.65292096  0.         -0.10047983 -0.30927835 -0.41237113 -0.65292096]


# from sarsa
pi = [0, 1, 0, 0, 0, 0, 3, 0, 3, 2, 3, 2]
V = get_state_values(pi, game.P)
print(V)

# [0.39473684 0.52631579 0.70175439 0.         0.29605263 0.
#  0.41569397 0.         0.22203947 0.21010187 0.26819822 0.20114866]

# vs sarsa estimate

# V is  [0.39415454 0.5264558  0.70510662 0.         0.29255546 0.
#        0.41554123 0.         0.22031379 0.2110992  0.27134116 0.19736997]

# from mc
pi = [0, 1, 0, 0, 3, 0, 3, 0, 0, 0, 1, 2]
V = get_state_values(pi, game.P)
print(V)

# [0.39473684 0.52631579 0.70175439 0.         0.25069324 0.
#  0.41237113 0.         0.19021406 0.19313957 0.26044493 0.19533369]

# vs mc estimate

# V is  [0.38753441 0.520514   0.69273459 0.         0.24347153 0.
#        0.400778   0.         0.1850012  0.18329231 0.24663445 0.18203998]

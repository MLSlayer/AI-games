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

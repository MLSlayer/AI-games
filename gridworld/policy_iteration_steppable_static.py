from gridworld.GridEnv import *

# the more typical policy iteration rhythm
def get_state_values(pi, P, gamma=0.9, theta=1e-10):
    V = np.zeros(len(pi))

    while True:
        delta = 0
        aux_V = np.zeros(len(pi))

        for s in range(len(P)):
            for prob, new_s, reward, is_done in P[s][pi[s]]:
                if is_done:
                    value = reward
                else:
                    value = reward + gamma * V[new_s]
                aux_V[s] += prob * value

            delta = max(delta, abs(aux_V[s] - V[s]))

        V = aux_V

        if delta < theta:
            break

    return V


def improve_policy(pi, V, P, nA, gamma=0.9):
    for s in range(len(P)):
        Qs = np.zeros(nA)

        for a in range(nA):
            for prob, new_s, reward, is_done in P[s][a]:
                if is_done:
                    Qs[a] += prob * reward
                else:
                    Qs[a] += prob * (reward + gamma * V[new_s])

        pi[s] = np.argmax(Qs)


game = GridEnv.steppable_static()
pi = [0, 1, 2, 3,
      0, 1, 2, 3,
      0, 1, 2, 3]

while True:
    prev_pi = pi.copy()
    V = get_state_values(pi, game.P)
    improve_policy(pi, V, game.P, game.nA)

    if prev_pi == pi:
        break

print(np.array(pi).reshape(3, 4))

# [[0 1 0 0]
#  [0 0 3 0]
#  [3 0 3 2]]


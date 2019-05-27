from gridworld.GridEnv import *

game = GridEnv.steppable_static()
P = game.P
nS = game.nS

# state value function memo
V = np.random.random(nS)

while True:
    delta = 0
    # action value function memo for all states
    QS = np.zeros((nS, game.nA))

    for s in range(nS):
        for a in range(game.nA):
            for prob, new_s, reward, is_done in P[s][a]:
                if is_done:
                    value = reward
                else:
                    value = reward + .9 * V[new_s]
                QS[s][a] += prob * value

        delta = max(delta, abs(np.max(QS[s]) - V[s]))
        V[s] = np.max(QS[s])

    if delta < 1e-10:
        break

pi = np.array([a for a in np.argmax(QS, axis=1)]).reshape(3,4)
print(pi)

# [[0 1 0 0]
#  [0 0 3 0]
#  [3 0 3 2]]


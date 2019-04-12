from gridworld.GridEnv import *

# GridEnv.static builds the below, p is the person, w is the wall, 1 and -1 are the terminal states
#       0  0  0  1
#       0  w  0 -1
#       0  0  0  0

game = GridEnv.static()
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
            new_s, reward, is_done = P[s][a]
            if is_done:
                value = reward
            else:
                value = reward + .9 * V[new_s]
            QS[s][a] += value

        delta = max(delta, abs(np.max(QS[s]) - V[s]))
        V[s] = np.max(QS[s])

    if delta < 1e-10:
        break

pi = np.array([a for a in np.argmax(QS, axis=1)]).reshape(3,4)
print(pi)

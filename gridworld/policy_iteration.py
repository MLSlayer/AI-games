from gridworld.GridEnv import *


# our policy is deterministic and our probablity of diff actions upon choosing the deterministic action is 0

def get_state_values(pi, P, gamma=0.9, theta=1e-10):
    V = np.zeros(len(pi))

    while True:
        delta = 0
        aux_V = np.zeros(len(pi))

        for s in range(len(P)):
            a = pi[s]
            new_s, reward, is_done = P[s][a]

            if is_done:
                value = reward
            else:
                value = reward + gamma * V[new_s]

            aux_V[s] += value
            delta = max(delta, abs(aux_V[s] - V[s]))

        V = aux_V

        if delta < theta:
            break

    return V


def improve_policy(pi, V, P, gamma=0.9):
    for s in range(len(V)):
        # action value function memo
        Qs = np.zeros(4)

        for a in range(4):
            new_s, reward, is_done = P[s][a]
            if is_done:
                value = reward
            else:
                value = reward + gamma * V[new_s]
            Qs[a] += value

        pi[s] = np.argmax(Qs)


# GridEnv.static builds the below, p is the person, w is the wall, 1 and -1 are the terminal states
#       0  0  0  1                                                u d l r
#       0  w  0 -1     actions/policies in client program ->      u d l r       where u,d,l,r = up,down,left,right
#       0  0  0  0                                                u d l r       also, w gives reward zero, but I just marked it w to show you its a wall

game = GridEnv.static()
pi = [0, 1, 2, 3,
      0, 1, 2, 3,
      0, 1, 2, 3]

while True:
    prev_pi = pi.copy()
    V = get_state_values(pi, game.P)
    improve_policy(pi, V, game.P)

    if prev_pi == pi:
        break

print(np.array(pi).reshape(3, 4))

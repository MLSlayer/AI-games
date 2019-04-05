from tictactoe.Game import *
import random
import torch as t
import pandas as pd

x_size = 9
a1_size = 81
a2_size = 45
a3_size = 9

model = t.nn.Sequential(
    t.nn.Linear(x_size, a1_size),
    t.nn.ReLU(),
    t.nn.Linear(a1_size, a2_size),
    t.nn.ReLU(),
    t.nn.Linear(a2_size, a3_size))

criterion = t.nn.MSELoss()
optimizer = t.optim.Adam(model.parameters())

pc_first_win_lose_confs = pd.read_csv("q_learning_meta/pc_first_win_lose_states.txt")
pc_first_win_lose_confs = dict(list(pc_first_win_lose_confs.itertuples(index=False, name=False)))

human_first_win_lose_confs = pd.read_csv("q_learning_meta/human_first_win_lose_states.txt")
human_first_win_lose_confs = dict(list(human_first_win_lose_confs.itertuples(index=False, name=False)))

epochs = 1500000
epsilon = .1
get_random_valid_move = lambda g: random.choice(list(g.valid_moves.keys()))
get_best_move = lambda q, valid_moves: next(e for e in map(lambda e: e[0], sorted(list(enumerate(q[0])), reverse=True, key=lambda e: e[1])) if e in valid_moves)

avg_losses = []
losses = 0
counter = 0
import time
start = time.time()
for epoch in range(epochs):
    x_first = random.random() < .5
    game = Game(X if x_first else O)
    if x_first:
        game.make_move(get_random_valid_move(game))

    state = t.Tensor(game.board.reshape(1, -1))
    y_turn = True
    while True:
        if y_turn:
            q = model(state)
            valid_moves = game.valid_moves.keys()
            if random.random() < epsilon:
                move = get_random_valid_move(game)
            else:
                move = get_best_move(q, valid_moves)

            game.make_move(move)
            board_to_str = lambda b: ''.join(map(lambda s: str(s), b))

            # only train weights associated with the for sure output.
            # technically we can train every weight and that's better, since we know the whole winning/lose config, but this is just to practice reinforcement rythm
            # and then maybe practice using batch/experience-replay to get around this fact just to practice reinforcement rythms..
            y = t.zeros(q.data.shape)
            y[:] = q.data[:]

            if x_first:
                reward = human_first_win_lose_confs[board_to_str(game.board.reshape(-1))]
            else:
                reward = pc_first_win_lose_confs[board_to_str(game.board.reshape(-1))]

            y[0][move] = reward
            loss = criterion(q, y)
            losses += loss.item()
            counter += 1
            if counter % 1000 == 0:
                avg_losses.append(str(losses / 1000))
                losses = 0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            y_turn = False
        else:
            game.make_move(get_random_valid_move(game))
            y_turn = True
        state = t.Tensor(game.board.reshape(1, -1))
        if game.is_over():
            break
    if epsilon > 0.01:
        epsilon -= 1 / epochs


t.save(model.state_dict(), "q_learning_meta/model.pth")
f = open("q_learning_meta/losses_over_time", "w")
f.write("\n".join(avg_losses))

end = time.time()
print(end - start)







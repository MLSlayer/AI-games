from tictactoe.Game import *
import random
import torch as t

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

x_first = random.random() < .5
game = Game(X if x_first else O)

model.load_state_dict(t.load("q_learning_meta/model.pth"))
get_best_move = lambda q, valid_moves: next(e for e in map(lambda e: e[0], sorted(list(enumerate(q[0])), reverse=True, key=lambda e: e[1])) if e in valid_moves)

if (x_first):
    print("valid moves are ", game.valid_moves)
    move = int(input("player one(X), make first move: "))
    game.make_move(move)

while not game.is_over():
    if game.player == X:
        print("valid moves are ", game.valid_moves)
        move = int(input("please make move: "))
    else:
        q = model(t.Tensor(game.board.reshape(1, -1)))
        print(q)
        move = get_best_move(q, game.valid_moves.keys())
        print("computer is now making move ", move)
    game.make_move(move)
    print(game)

if game.winner_exists():
    print("You win!" if game.get_winner() == X else "You lose!")
else:
    print("player one and two ties")


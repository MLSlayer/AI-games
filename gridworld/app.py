from Grid import *

game = Grid(True)
game.print_grid()
game.move_down()
game.print_grid()
game.move_up()
game.print_grid()
game.move_right()
game.print_grid()
game.move_up()
game.print_grid()
game.move_left()
game.print_grid()
game.move_up()
game.print_grid()
game.move_right()
game.print_grid()
game.move_right()
game.print_grid()
game.move_right()
game.print_grid()
print(game.is_over)
try:
    game.move_right()
    print("doesn't hit this")
except AssertionError as e:
    print("hits this")


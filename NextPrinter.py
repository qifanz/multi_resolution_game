from math import floor
import numpy as np
import sys
from Game import Game

state = 37
action1 = 1
action2 = 0



if __name__ == '__main__':
    state = int(sys.argv[1])
    action1 = int(sys.argv[2])
    action2 = int(sys.argv[3])

    game = Game()
    next_state = game.get_next_state(state, action1, action2)
    print(next_state)
    print('---------')
    game.print_state(next_state)
